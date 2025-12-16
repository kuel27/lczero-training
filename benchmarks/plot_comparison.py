#!/usr/bin/env python3
"""Plot training comparison results.

Creates visualizations from the training comparison output.

Usage:
    uv run python benchmarks/plot_comparison.py benchmarks/results/comparison_results.json
    uv run python benchmarks/plot_comparison.py benchmarks/results/comparison_results.json --output plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(results_path: Path) -> list[dict]:
    """Load results from JSON file."""
    return json.loads(results_path.read_text(encoding="utf-8"))


def plot_loss_curves(results: list[dict], output_dir: Path) -> None:
    """Plot loss curves for all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color map for configurations
    colors = {
        "baseline": "#1f77b4",
        "rmsnorm_only": "#ff7f0e",
        "rope_only": "#2ca02c",
        "swiglu_only": "#d62728",
        "all_features": "#9467bd",
    }

    # Style map
    styles = {
        "baseline": "-",
        "rmsnorm_only": "--",
        "rope_only": "-.",
        "swiglu_only": ":",
        "all_features": "-",
    }

    for result in results:
        name = result["config_name"]
        history = result["step_history"]
        if not history:
            continue

        steps = [h["step"] for h in history]
        color = colors.get(name, "#333333")
        style = styles.get(name, "-")
        lw = 2.5 if name in ("baseline", "all_features") else 1.5

        # Total loss
        total_loss = [h["total_loss"] for h in history]
        axes[0, 0].plot(steps, total_loss, label=name, color=color,
                       linestyle=style, linewidth=lw)

        # Policy loss
        policy_loss = [h["policy_loss"] for h in history]
        axes[0, 1].plot(steps, policy_loss, label=name, color=color,
                       linestyle=style, linewidth=lw)

        # Value loss
        value_loss = [h["value_loss"] for h in history]
        axes[1, 0].plot(steps, value_loss, label=name, color=color,
                       linestyle=style, linewidth=lw)

        # Moves-left loss
        movesleft_loss = [h["movesleft_loss"] for h in history]
        axes[1, 1].plot(steps, movesleft_loss, label=name, color=color,
                       linestyle=style, linewidth=lw)

    # Configure subplots
    titles = ["Total Loss", "Policy Loss", "Value Loss", "Moves-Left Loss"]
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    plt.suptitle("Training Loss Comparison: Feature Ablation Study", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_comparison(results: list[dict], output_dir: Path) -> None:
    """Plot bar chart comparing final losses."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Sort by total loss
    sorted_results = sorted(results, key=lambda r: r["final_total_loss"])
    names = [r["config_name"] for r in sorted_results]

    # Colors based on features
    def get_color(r: dict) -> str:
        if r["normalization"] == "RMSNorm" and r["ffn_activation"] == "SwiGLU" and r["use_rope"]:
            return "#9467bd"  # all features - purple
        elif r["config_name"] == "baseline":
            return "#1f77b4"  # baseline - blue
        elif r["normalization"] == "RMSNorm":
            return "#ff7f0e"  # rmsnorm - orange
        elif r["use_rope"]:
            return "#2ca02c"  # rope - green
        elif r["ffn_activation"] == "SwiGLU":
            return "#d62728"  # swiglu - red
        return "#333333"

    colors = [get_color(r) for r in sorted_results]

    # Total loss
    total_losses = [r["final_total_loss"] for r in sorted_results]
    axes[0].barh(names, total_losses, color=colors)
    axes[0].set_xlabel("Final Total Loss")
    axes[0].set_title("Total Loss (lower is better)")
    axes[0].invert_yaxis()

    # Policy loss
    policy_losses = [r["final_policy_loss"] for r in sorted_results]
    axes[1].barh(names, policy_losses, color=colors)
    axes[1].set_xlabel("Final Policy Loss")
    axes[1].set_title("Policy Loss (lower is better)")
    axes[1].invert_yaxis()

    # Value loss
    value_losses = [r["final_value_loss"] for r in sorted_results]
    axes[2].barh(names, value_losses, color=colors)
    axes[2].set_xlabel("Final Value Loss")
    axes[2].set_title("Value Loss (lower is better)")
    axes[2].invert_yaxis()

    plt.suptitle("Final Loss Comparison (Sorted by Total Loss)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "final_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_improvement_over_baseline(results: list[dict], output_dir: Path) -> None:
    """Plot percentage improvement over baseline."""
    baseline = next((r for r in results if r["config_name"] == "baseline"), None)
    if baseline is None:
        print("No baseline found, skipping improvement plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    other_results = [r for r in results if r["config_name"] != "baseline"]

    # Calculate improvements
    names = []
    improvements = []
    colors = []

    color_map = {
        "rmsnorm_only": "#ff7f0e",
        "rope_only": "#2ca02c",
        "swiglu_only": "#d62728",
        "all_features": "#9467bd",
    }

    for r in other_results:
        improvement = ((baseline["final_total_loss"] - r["final_total_loss"])
                      / baseline["final_total_loss"] * 100)
        names.append(r["config_name"])
        improvements.append(improvement)
        colors.append(color_map.get(r["config_name"], "#333333"))

    # Sort by improvement
    sorted_data = sorted(zip(names, improvements, colors), key=lambda x: x[1], reverse=True)
    names, improvements, colors = zip(*sorted_data) if sorted_data else ([], [], [])

    bars = ax.barh(list(names), list(improvements), color=list(colors))

    # Add value labels
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        label_x = width + 0.3 if width >= 0 else width - 0.3
        ha = "left" if width >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{imp:.2f}%", va="center", ha=ha, fontsize=10)

    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Improvement over Baseline (%)")
    ax.set_title("Loss Improvement vs Baseline (positive = better)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "improvement_over_baseline.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_table(results: list[dict], output_dir: Path) -> None:
    """Create a convergence comparison table as an image."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Prepare data
    headers = ["Config", "Norm", "FFN", "RoPE", "@1k", "@2k", "@5k", "@10k", "Final"]
    sorted_results = sorted(results, key=lambda r: r["final_total_loss"])

    cell_data = []
    for r in sorted_results:
        row = [
            r["config_name"],
            r["normalization"],
            r["ffn_activation"],
            "Yes" if r["use_rope"] else "No",
            f"{r['loss_at_1k']:.4f}" if r.get("loss_at_1k") else "N/A",
            f"{r['loss_at_2k']:.4f}" if r.get("loss_at_2k") else "N/A",
            f"{r['loss_at_5k']:.4f}" if r.get("loss_at_5k") else "N/A",
            f"{r['loss_at_10k']:.4f}" if r.get("loss_at_10k") else "N/A",
            f"{r['final_total_loss']:.4f}",
        ]
        cell_data.append(row)

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Highlight best row
    for i in range(len(headers)):
        table[(1, i)].set_facecolor("#E2EFDA")

    plt.title("Convergence Comparison (Loss at Various Steps)", fontsize=12, fontweight="bold", pad=20)

    output_path = output_dir / "convergence_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_markdown_report(results: list[dict], output_dir: Path) -> None:
    """Generate a markdown report with results."""
    sorted_results = sorted(results, key=lambda r: r["final_total_loss"])
    baseline = next((r for r in results if r["config_name"] == "baseline"), None)

    lines = [
        "# Training Comparison Results",
        "",
        "## Summary",
        "",
        "| Config | Norm | FFN | RoPE | Params | Final Loss | Policy | Value |",
        "|--------|------|-----|------|--------|------------|--------|-------|",
    ]

    for r in sorted_results:
        lines.append(
            f"| {r['config_name']} | {r['normalization']} | {r['ffn_activation']} | "
            f"{'Yes' if r['use_rope'] else 'No'} | {r['num_params']:,} | "
            f"{r['final_total_loss']:.4f} | {r['final_policy_loss']:.4f} | "
            f"{r['final_value_loss']:.4f} |"
        )

    lines.extend([
        "",
        "## Best Configuration",
        "",
        f"**{sorted_results[0]['config_name']}** achieved the lowest loss.",
        "",
    ])

    if baseline and sorted_results[0]["config_name"] != "baseline":
        improvement = ((baseline["final_total_loss"] - sorted_results[0]["final_total_loss"])
                      / baseline["final_total_loss"] * 100)
        lines.append(f"Improvement over baseline: **{improvement:.2f}%**")
        lines.append("")

    lines.extend([
        "## Convergence",
        "",
        "| Config | @1k | @2k | @5k | @10k |",
        "|--------|-----|-----|-----|------|",
    ])

    for r in sorted_results:
        def fmt_loss(val):
            return f"{val:.4f}" if val is not None else "N/A"

        lines.append(
            f"| {r['config_name']} | "
            f"{fmt_loss(r.get('loss_at_1k'))} | "
            f"{fmt_loss(r.get('loss_at_2k'))} | "
            f"{fmt_loss(r.get('loss_at_5k'))} | "
            f"{fmt_loss(r.get('loss_at_10k'))} |"
        )

    lines.extend([
        "",
        "## Plots",
        "",
        "![Loss Curves](loss_curves.png)",
        "",
        "![Final Comparison](final_comparison.png)",
        "",
        "![Improvement](improvement_over_baseline.png)",
        "",
    ])

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot training comparison results")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to comparison_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results file)",
    )

    args = parser.parse_args(argv)
    results_path = Path(args.results_file)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1

    output_dir = Path(args.output) if args.output else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)

    if not results:
        print("Error: No results found in file")
        return 1

    print(f"Loaded {len(results)} training results")

    if HAS_MATPLOTLIB:
        plot_loss_curves(results, output_dir)
        plot_final_comparison(results, output_dir)
        plot_improvement_over_baseline(results, output_dir)
        plot_convergence_table(results, output_dir)
    else:
        print("Warning: matplotlib not installed, skipping plots")
        print("Install with: uv pip install matplotlib")

    generate_markdown_report(results, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
