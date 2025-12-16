#!/bin/bash
# Run all ablation benchmarks for lczero-training fork
#
# Usage:
#   ./benchmarks/run_benchmarks.sh              # Run all benchmarks
#   ./benchmarks/run_benchmarks.sh --quick      # Quick run with fewer steps
#   ./benchmarks/run_benchmarks.sh baseline     # Run single config

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default settings
WARMUP_STEPS=10
BENCHMARK_STEPS=50
BATCH_SIZE=64
OUTPUT_DIR="$PROJECT_ROOT/benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/benchmark_results_$TIMESTAMP.json"

# Parse arguments
QUICK_MODE=false
SINGLE_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            WARMUP_STEPS=3
            BENCHMARK_STEPS=10
            shift
            ;;
        --warmup)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --steps)
            BENCHMARK_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            SINGLE_CONFIG="$1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "LCZero Training Fork Benchmark Suite"
echo "=============================================="
echo "Warmup steps: $WARMUP_STEPS"
echo "Benchmark steps: $BENCHMARK_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_FILE"
echo "=============================================="
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Run benchmarks
if [[ -n "$SINGLE_CONFIG" ]]; then
    # Single config mode
    CONFIG_PATH="$SCRIPT_DIR/configs/${SINGLE_CONFIG}.textproto"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Error: Config not found: $CONFIG_PATH"
        exit 1
    fi
    echo "Running benchmark for: $SINGLE_CONFIG"
    uv run python "$SCRIPT_DIR/benchmark.py" \
        --config "$CONFIG_PATH" \
        --warmup-steps "$WARMUP_STEPS" \
        --benchmark-steps "$BENCHMARK_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --output "$OUTPUT_FILE"
else
    # Run all benchmarks
    echo "Running all ablation benchmarks..."
    uv run python "$SCRIPT_DIR/benchmark.py" \
        --all \
        --configs-dir "$SCRIPT_DIR/configs" \
        --warmup-steps "$WARMUP_STEPS" \
        --benchmark-steps "$BENCHMARK_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --output "$OUTPUT_FILE"
fi

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=============================================="
