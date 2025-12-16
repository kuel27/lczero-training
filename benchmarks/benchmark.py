#!/usr/bin/env python3
"""Benchmark script for comparing lczero-training feature configurations.

This script measures:
- Training throughput (steps/second)
- Memory usage (peak GPU memory)
- Forward/backward pass latency

Usage:
    uv run python benchmarks/benchmark.py --config benchmarks/configs/baseline.textproto
    uv run python benchmarks/benchmark.py --all  # Run all ablation configs
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from google.protobuf import text_format

# Configure logging before imports that might log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config_name: str
    normalization: str
    ffn_activation: str
    use_rope: bool
    num_params: int
    warmup_steps: int
    benchmark_steps: int
    total_time_seconds: float
    steps_per_second: float
    avg_step_time_ms: float
    peak_memory_bytes: Optional[int]
    peak_memory_mb: Optional[float]


def count_parameters(model: nnx.Module) -> int:
    """Count total trainable parameters in a model."""
    _, state = nnx.split(model)
    total = 0
    for leaf in jax.tree_util.tree_leaves(state):
        if hasattr(leaf, "value") and hasattr(leaf.value, "size"):
            total += leaf.value.size
        elif hasattr(leaf, "size"):
            total += leaf.size
    return total


def get_peak_memory() -> Optional[int]:
    """Get peak GPU memory usage in bytes."""
    try:
        backend = jax.lib.xla_bridge.get_backend()
        if hasattr(backend, "live_buffers"):
            buffers = backend.live_buffers()
            return sum(b.nbytes for b in buffers)
    except Exception:
        pass
    return None


def run_benchmark(
    config_path: str,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
    batch_size: int = 64,
) -> BenchmarkResult:
    """Run benchmark on a single configuration.

    Args:
        config_path: Path to the textproto configuration file.
        warmup_steps: Number of warmup steps before timing.
        benchmark_steps: Number of steps to time.
        batch_size: Batch size for synthetic data.

    Returns:
        BenchmarkResult with timing and memory statistics.
    """
    # Import here to avoid loading JAX before argument parsing
    from lczero_training.model.loss_function import LczeroLoss
    from lczero_training.model.model import LczeroModel
    from lczero_training.training.lr_schedule import make_lr_schedule
    from lczero_training.training.optimizer import make_gradient_transformation
    from proto import model_config_pb2, net_pb2
    from proto.root_config_pb2 import RootConfig

    logger.info(f"Loading configuration from {config_path}")
    config = RootConfig()
    with open(config_path, "r") as f:
        text_format.Parse(f.read(), config)

    config_name = Path(config_path).stem

    # Extract feature flags for reporting
    defaults = config.model.defaults
    norm_type = "RMSNorm" if defaults.normalization == model_config_pb2.RMS_NORM else "LayerNorm"
    ffn_type = (
        "SwiGLU"
        if defaults.ffn_activation == net_pb2.NetworkFormat.ACTIVATION_SWIGLU
        else "MISH"
    )
    use_rope = config.model.encoder.use_rope

    logger.info(f"Config: {config_name}")
    logger.info(f"  Normalization: {norm_type}")
    logger.info(f"  FFN Activation: {ffn_type}")
    logger.info(f"  RoPE: {use_rope}")

    # Create model
    logger.info("Creating model...")
    rngs = nnx.Rngs(params=42)
    model = LczeroModel(config=config.model, rngs=rngs)
    num_params = count_parameters(model)
    logger.info(f"  Parameters: {num_params:,}")

    # Create optimizer
    lr_sched = make_lr_schedule(config.training.lr_schedule)
    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
        lr_schedule=lr_sched,
    )

    # Split model for training
    graphdef, state = nnx.split(model)
    opt_state = optimizer_tx.init(state)

    # Create loss function
    loss_fn = LczeroLoss(config=config.training.losses)

    # Create synthetic training data
    key = jax.random.key(0)

    def make_batch(key: jax.Array) -> dict:
        """Generate a synthetic training batch."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return {
            "input_planes": jax.random.normal(k1, (batch_size, 112, 8, 8)),
            "policy_target": jax.nn.softmax(
                jax.random.normal(k2, (batch_size, 1858)), axis=-1
            ),
            "value_target": jax.random.uniform(k3, (batch_size, 3)),
            "movesleft_target": jax.random.uniform(k4, (batch_size, 1)) * 100,
            "legal_moves_mask": jax.random.bernoulli(k5, 0.1, (batch_size, 1858)),
        }

    @jax.jit
    def train_step(
        state: nnx.State,
        opt_state: nnx.State,
        batch: dict,
    ) -> tuple[nnx.State, nnx.State, jax.Array]:
        """Single training step."""

        def loss_wrapper(state: nnx.State) -> jax.Array:
            model = nnx.merge(graphdef, state)

            def loss_for_grad(
                model_arg: LczeroModel, batch_arg: dict
            ) -> jax.Array:
                loss, _ = loss_fn(
                    model_arg,
                    inputs=batch_arg["input_planes"],
                    value_targets=batch_arg["value_target"],
                    policy_targets=batch_arg["policy_target"],
                    movesleft_targets=batch_arg["movesleft_target"],
                )
                return loss

            # vmap over batch dimension
            loss_vfn = jax.vmap(
                loss_for_grad,
                in_axes=(None, 0),
                out_axes=0,
            )
            per_sample_loss = loss_vfn(model, batch)
            return jnp.mean(per_sample_loss)

        loss, grads = jax.value_and_grad(loss_wrapper)(state)
        updates, new_opt_state = optimizer_tx.update(grads, opt_state, state)
        new_state = jax.tree_util.tree_map(
            lambda p, u: p + u, state, updates
        )
        return new_state, new_opt_state, loss

    # Warmup
    logger.info(f"Running {warmup_steps} warmup steps...")
    for i in range(warmup_steps):
        key, subkey = jax.random.split(key)
        batch = make_batch(subkey)
        state, opt_state, loss = train_step(state, opt_state, batch)
        # Block to ensure warmup completes
        loss.block_until_ready()

    # Clear caches
    gc.collect()

    # Benchmark
    logger.info(f"Running {benchmark_steps} benchmark steps...")
    start_time = time.perf_counter()

    for i in range(benchmark_steps):
        key, subkey = jax.random.split(key)
        batch = make_batch(subkey)
        state, opt_state, loss = train_step(state, opt_state, batch)

    # Ensure all async operations complete
    loss.block_until_ready()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    steps_per_second = benchmark_steps / total_time
    avg_step_time_ms = (total_time / benchmark_steps) * 1000

    # Get memory usage
    peak_memory = get_peak_memory()
    peak_memory_mb = peak_memory / (1024 * 1024) if peak_memory else None

    logger.info(f"Results for {config_name}:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Steps/second: {steps_per_second:.2f}")
    logger.info(f"  Avg step time: {avg_step_time_ms:.2f}ms")
    if peak_memory_mb:
        logger.info(f"  Peak memory: {peak_memory_mb:.2f}MB")

    return BenchmarkResult(
        config_name=config_name,
        normalization=norm_type,
        ffn_activation=ffn_type,
        use_rope=use_rope,
        num_params=num_params,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        total_time_seconds=total_time,
        steps_per_second=steps_per_second,
        avg_step_time_ms=avg_step_time_ms,
        peak_memory_bytes=peak_memory,
        peak_memory_mb=peak_memory_mb,
    )


def run_all_benchmarks(
    configs_dir: str = "benchmarks/configs",
    output_file: Optional[str] = None,
    **kwargs,
) -> list[BenchmarkResult]:
    """Run benchmarks on all configuration files in a directory."""
    configs_path = Path(configs_dir)
    config_files = sorted(configs_path.glob("*.textproto"))

    if not config_files:
        logger.error(f"No .textproto files found in {configs_dir}")
        return []

    results = []
    for config_file in config_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {config_file.name}")
        logger.info(f"{'='*60}")

        try:
            result = run_benchmark(str(config_file), **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {config_file.name}: {e}")
            continue

        # Clear GPU memory between runs
        gc.collect()

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Config':<20} {'Norm':<10} {'FFN':<8} {'RoPE':<6} "
        f"{'Params':<12} {'Steps/s':<10} {'ms/step':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.config_name:<20} {r.normalization:<10} {r.ffn_activation:<8} "
            f"{str(r.use_rope):<6} {r.num_params:<12,} {r.steps_per_second:<10.2f} "
            f"{r.avg_step_time_ms:<10.2f}"
        )

    print("=" * 80)

    # Save results to JSON if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark lczero-training configurations"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a single configuration file to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run benchmarks on all configs in benchmarks/configs/",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="benchmarks/configs",
        help="Directory containing configuration files (default: benchmarks/configs)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps (default: 10)",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=50,
        help="Number of steps to benchmark (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for synthetic data (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("Either --config or --all must be specified")

    benchmark_kwargs = {
        "warmup_steps": args.warmup_steps,
        "benchmark_steps": args.benchmark_steps,
        "batch_size": args.batch_size,
    }

    if args.all:
        results = run_all_benchmarks(
            configs_dir=args.configs_dir,
            output_file=args.output,
            **benchmark_kwargs,
        )
        return 0 if results else 1
    else:
        result = run_benchmark(args.config, **benchmark_kwargs)
        if args.output:
            with open(args.output, "w") as f:
                json.dump([asdict(result)], f, indent=2)
        return 0


if __name__ == "__main__":
    sys.exit(main())
