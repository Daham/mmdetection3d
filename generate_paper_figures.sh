#!/bin/bash
# Quick script to generate qualitative comparison figures for research paper

# Activate virtual environment
source /home/daham/mmdetection_project/mmdet_env/bin/activate

echo "=========================================="
echo "Generating Qualitative Comparison Figures"
echo "=========================================="

# Default paths
BASELINE_CONFIG="configs/second/validation_baseline_01_single_scale_80ep.py"
BASELINE_CKPT="work_dirs/comparison_5epochs/method1_single/epoch_5.pth"
VOXADAPT_CONFIG="configs/second/baseline_03_adaptive_multiscale_learnable.py"
VOXADAPT_CKPT="work_dirs/method3_with_fix_5epochs/epoch_5.pth"
DATA_ROOT="/home/daham/mmdetection_project/dataset/KITTI"
OUTPUT_DIR="qualitative_results"
NUM_SAMPLES=6
SCORE_THR=0.3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --score-thr)
            SCORE_THR="$2"
            shift 2
            ;;
        --sample-indices)
            shift
            SAMPLE_INDICES="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num-samples N] [--output-dir DIR] [--score-thr THR] [--sample-indices IDX1 IDX2 ...]"
            exit 1
            ;;
    esac
done

# Build command
CMD="python generate_qualitative_comparison.py \
    --baseline-config $BASELINE_CONFIG \
    --baseline-checkpoint $BASELINE_CKPT \
    --voxadapt-config $VOXADAPT_CONFIG \
    --voxadapt-checkpoint $VOXADAPT_CKPT \
    --data-root $DATA_ROOT \
    --num-samples $NUM_SAMPLES \
    --output-dir $OUTPUT_DIR \
    --score-thr $SCORE_THR \
    --device cuda:0"

if [ -n "$SAMPLE_INDICES" ]; then
    CMD="$CMD --sample-indices $SAMPLE_INDICES"
fi

echo "Running: $CMD"
echo ""

# Run the script
eval $CMD

echo ""
echo "Done! Check $OUTPUT_DIR/ for results."
