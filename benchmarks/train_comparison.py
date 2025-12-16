#!/usr/bin/env python3
"""Automated training comparison for lczero-training feature ablation study.

This script trains multiple configurations on real data and compares:
- Policy loss (move prediction accuracy)
- Value loss (win/draw/loss prediction)
- Moves-left loss
- Convergence speed (loss at various checkpoints)

Usage:
    uv run python benchmarks/train_comparison.py --steps 5000
    uv run python benchmarks/train_comparison.py --steps 10000 --configs baseline all_features
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from google.protobuf import text_format
from jax import tree_util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics captured at each training step."""

    step: int
    total_loss: float
    policy_loss: float
    value_loss: float
    movesleft_loss: float
    grad_norm: float
    elapsed_seconds: float


@dataclass
class TrainingResult:
    """Complete results from a training run."""

    config_name: str
    normalization: str
    ffn_activation: str
    use_rope: bool
    num_params: int
    total_steps: int
    total_time_seconds: float
    final_total_loss: float
    final_policy_loss: float
    final_value_loss: float
    final_movesleft_loss: float
    # Loss at various checkpoints for convergence analysis
    loss_at_1k: float | None = None
    loss_at_2k: float | None = None
    loss_at_5k: float | None = None
    loss_at_10k: float | None = None
    # Step history for plotting
    step_history: list[StepMetrics] = field(default_factory=list)


def load_config(config_path: Path) -> Any:
    """Load a textproto configuration file."""
    from proto.root_config_pb2 import RootConfig

    config = RootConfig()
    config_text = config_path.read_text(encoding="utf-8")
    text_format.Parse(config_text, config)
    return config


def extract_features(config: Any) -> dict[str, Any]:
    """Extract feature flags from config for reporting."""
    from proto import model_config_pb2, net_pb2

    defaults = config.model.defaults

    norm_map = {model_config_pb2.RMS_NORM: "RMSNorm"}
    normalization = norm_map.get(defaults.normalization, "LayerNorm")

    ffn_map = {net_pb2.NetworkFormat.ACTIVATION_SWIGLU: "SwiGLU"}
    ffn_activation = ffn_map.get(defaults.ffn_activation, "MISH")

    use_rope = bool(config.model.encoder.use_rope)

    return {
        "normalization": normalization,
        "ffn_activation": ffn_activation,
        "use_rope": use_rope,
    }


def count_parameters(model: Any) -> int:
    """Count total trainable parameters in a model."""
    _, state = nnx.split(model)
    total = 0
    for leaf in jax.tree_util.tree_leaves(state):
        if hasattr(leaf, "value") and hasattr(leaf.value, "size"):
            total += int(leaf.value.size)
        elif hasattr(leaf, "size"):
            total += int(leaf.size)
    return total


def run_training(
    config_path: Path,
    num_steps: int,
    log_interval: int = 100,
    output_dir: Path | None = None,
) -> TrainingResult:
    """Run training on a single configuration and collect metrics.

    Args:
        config_path: Path to the .textproto config file
        num_steps: Number of training steps to run
        log_interval: How often to record metrics (every N steps)
        output_dir: Optional directory for CSV output

    Returns:
        TrainingResult with all metrics
    """
    from lczero_training.dataloader import make_dataloader
    from lczero_training.model.loss_function import LczeroLoss
    from lczero_training.model.model import LczeroModel
    from lczero_training.training.lr_schedule import make_lr_schedule
    from lczero_training.training.optimizer import make_gradient_transformation
    from lczero_training.training.state import TrainingState
    from lczero_training.training.training import Training

    config_name = config_path.stem
    logger.info("=" * 60)
    logger.info(f"Training: {config_name}")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    features = extract_features(config)

    logger.info(f"  Normalization: {features['normalization']}")
    logger.info(f"  FFN Activation: {features['ffn_activation']}")
    logger.info(f"  RoPE: {features['use_rope']}")

    # Create fresh training state (random init)
    logger.info("Initializing model with random weights...")
    training_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    # Get model graphdef and count params
    model_instance = LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    num_params = count_parameters(model_instance)
    graphdef, _ = nnx.split(model_instance)
    del model_instance

    logger.info(f"  Parameters: {num_params:,}")

    # Setup optimizer and training
    jit_state = training_state.jit_state
    lr_sched = make_lr_schedule(config.training.lr_schedule)
    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
        lr_schedule=lr_sched,
    )

    loss_fn = LczeroLoss(config=config.training.losses)
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=graphdef,
        loss_fn=loss_fn,
    )

    # Create dataloader
    logger.info("Creating data loader...")
    loader = make_dataloader(config.data_loader)

    # Prepare CSV output
    csv_file = None
    csv_writer = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{config_name}_metrics.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step", "total_loss", "policy_loss", "value_loss",
            "movesleft_loss", "grad_norm", "elapsed_seconds"
        ])

    # Training loop
    step_history: list[StepMetrics] = []
    start_time = time.perf_counter()

    logger.info(f"Starting training for {num_steps} steps...")

    try:
        for step in range(num_steps):
            # Get batch from dataloader
            batch_tuple = loader.get_next("train")
            inputs, policy, values, _, movesleft = batch_tuple

            batch_dict = {
                "inputs": inputs,
                "value_targets": values,
                "policy_targets": policy,
                "movesleft_targets": movesleft,
            }

            # Execute training step
            jit_state, metrics = training.train_step(
                optimizer_tx, jit_state, batch_dict
            )

            # Extract metrics
            loss = float(np.asarray(jax.device_get(metrics["loss"])))
            grad_norm = float(np.asarray(jax.device_get(metrics["grad_norm"])))
            unweighted = jax.device_get(metrics["unweighted_losses"])
            policy_loss = float(np.asarray(unweighted["policy"]))
            value_loss = float(np.asarray(unweighted["value"]))
            movesleft_loss = float(np.asarray(unweighted["movesleft"]))
            elapsed = time.perf_counter() - start_time

            current_step = step + 1

            # Log at intervals
            if current_step % log_interval == 0 or current_step == num_steps:
                step_metrics = StepMetrics(
                    step=current_step,
                    total_loss=loss,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    movesleft_loss=movesleft_loss,
                    grad_norm=grad_norm,
                    elapsed_seconds=elapsed,
                )
                step_history.append(step_metrics)

                logger.info(
                    f"Step {current_step}/{num_steps}: "
                    f"loss={loss:.4f} (p={policy_loss:.4f}, v={value_loss:.4f}, m={movesleft_loss:.4f}) "
                    f"grad={grad_norm:.4f} [{elapsed:.1f}s]"
                )

                if csv_writer is not None:
                    csv_writer.writerow([
                        current_step, loss, policy_loss, value_loss,
                        movesleft_loss, grad_norm, elapsed
                    ])
                    csv_file.flush()

    finally:
        if csv_file is not None:
            csv_file.close()
        # Stop the dataloader
        try:
            loader.stop()
        except Exception:
            pass

    total_time = time.perf_counter() - start_time

    # Extract checkpoint losses
    def get_loss_at_step(target_step: int) -> float | None:
        for m in step_history:
            if m.step >= target_step:
                return m.total_loss
        return None

    result = TrainingResult(
        config_name=config_name,
        normalization=features["normalization"],
        ffn_activation=features["ffn_activation"],
        use_rope=features["use_rope"],
        num_params=num_params,
        total_steps=num_steps,
        total_time_seconds=total_time,
        final_total_loss=step_history[-1].total_loss if step_history else 0.0,
        final_policy_loss=step_history[-1].policy_loss if step_history else 0.0,
        final_value_loss=step_history[-1].value_loss if step_history else 0.0,
        final_movesleft_loss=step_history[-1].movesleft_loss if step_history else 0.0,
        loss_at_1k=get_loss_at_step(1000),
        loss_at_2k=get_loss_at_step(2000),
        loss_at_5k=get_loss_at_step(5000),
        loss_at_10k=get_loss_at_step(10000),
        step_history=step_history,
    )

    logger.info(f"Completed {config_name} in {total_time:.1f}s")
    logger.info(f"  Final loss: {result.final_total_loss:.4f}")

    return result


def print_comparison_table(results: list[TrainingResult]) -> None:
    """Print a comparison table of all results."""
    print("\n" + "=" * 100)
    print("TRAINING COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(
        f"{'Config':<20} {'Norm':<10} {'FFN':<8} {'RoPE':<6} "
        f"{'Params':<12} {'Final Loss':<12} {'Policy':<10} {'Value':<10} {'Time':<10}"
    )
    print("-" * 100)

    # Sort by final loss
    sorted_results = sorted(results, key=lambda r: r.final_total_loss)

    for r in sorted_results:
        print(
            f"{r.config_name:<20} {r.normalization:<10} {r.ffn_activation:<8} "
            f"{str(r.use_rope):<6} {r.num_params:<12,} {r.final_total_loss:<12.4f} "
            f"{r.final_policy_loss:<10.4f} {r.final_value_loss:<10.4f} "
            f"{r.total_time_seconds:<10.1f}s"
        )

    print("=" * 100)

    # Convergence comparison
    print("\nCONVERGENCE COMPARISON (Total Loss at Step N)")
    print("-" * 80)
    print(f"{'Config':<20} {'@1k':<12} {'@2k':<12} {'@5k':<12} {'@10k':<12}")
    print("-" * 80)

    for r in sorted_results:
        loss_1k = f"{r.loss_at_1k:.4f}" if r.loss_at_1k else "N/A"
        loss_2k = f"{r.loss_at_2k:.4f}" if r.loss_at_2k else "N/A"
        loss_5k = f"{r.loss_at_5k:.4f}" if r.loss_at_5k else "N/A"
        loss_10k = f"{r.loss_at_10k:.4f}" if r.loss_at_10k else "N/A"
        print(f"{r.config_name:<20} {loss_1k:<12} {loss_2k:<12} {loss_5k:<12} {loss_10k:<12}")

    print("=" * 80)

    # Best configuration
    best = sorted_results[0]
    baseline = next((r for r in results if r.config_name == "baseline"), None)

    print(f"\nBEST CONFIGURATION: {best.config_name}")
    print(f"  Final Loss: {best.final_total_loss:.4f}")

    if baseline and baseline.config_name != best.config_name:
        improvement = ((baseline.final_total_loss - best.final_total_loss)
                      / baseline.final_total_loss * 100)
        print(f"  Improvement over baseline: {improvement:.2f}%")


def save_results(results: list[TrainingResult], output_path: Path) -> None:
    """Save results to JSON file."""
    # Convert to serializable format (exclude step_history for main file)
    data = []
    for r in results:
        d = asdict(r)
        # Convert step_history to simple dicts
        d["step_history"] = [asdict(s) for s in r.step_history]
        data.append(d)

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {output_path}")


def run_all_configs(
    configs_dir: Path,
    num_steps: int,
    log_interval: int,
    output_dir: Path,
    config_filter: list[str] | None = None,
) -> list[TrainingResult]:
    """Run training on all configurations."""
    config_files = sorted(configs_dir.glob("*.textproto"))

    if config_filter:
        config_files = [
            f for f in config_files
            if f.stem in config_filter
        ]

    if not config_files:
        logger.error(f"No config files found in {configs_dir}")
        return []

    logger.info(f"Found {len(config_files)} configurations to train")
    for f in config_files:
        logger.info(f"  - {f.stem}")

    results: list[TrainingResult] = []

    for config_file in config_files:
        try:
            result = run_training(
                config_path=config_file,
                num_steps=num_steps,
                log_interval=log_interval,
                output_dir=output_dir,
            )
            results.append(result)
        except Exception as e:
            logger.exception(f"Failed to train {config_file.name}: {e}")

        # Clean up between runs
        gc.collect()

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run training comparison across lczero-training configurations"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps per configuration (default: 5000)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log metrics every N steps (default: 100)",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="benchmarks/configs",
        help="Directory containing config files (default: benchmarks/configs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory for output files (default: benchmarks/results)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Specific config names to run (e.g., baseline all_features)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before running",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)

    if args.clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {args.steps} steps per configuration")
    logger.info(f"Output directory: {output_dir}")

    results = run_all_configs(
        configs_dir=configs_dir,
        num_steps=args.steps,
        log_interval=args.log_interval,
        output_dir=output_dir,
        config_filter=args.configs,
    )

    if not results:
        logger.error("No training runs completed successfully")
        return 1

    # Print comparison
    print_comparison_table(results)

    # Save results
    results_file = output_dir / "comparison_results.json"
    save_results(results, results_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
