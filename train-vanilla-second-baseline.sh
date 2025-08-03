#!/bin/bash

echo "===================================================================================="
echo "PhD RESEARCH BENCHMARK: Training Vanilla SECOND Baseline"
echo "===================================================================================="
echo "Purpose: Establish baseline performance for comparison with adaptive voxelization"
echo "Configuration: Standard SECOND with fixed voxel size (0.05, 0.05, 0.1)"
echo "Expected Performance: ~0.5-1.0s/iter (typical SECOND performance)"
echo ""

# Set working directory
cd /home/daham/mmdetection_project/mmdetection3d

# Clean previous baseline results
if [ -d "./work_dirs/vanilla_second_baseline" ]; then
    echo "Cleaning previous baseline results..."
    rm -rf ./work_dirs/vanilla_second_baseline
fi

echo "Starting vanilla SECOND baseline training..."
echo "Configuration: configs/vanilla_second_baseline.py"
echo ""

# Training command
python tools/train.py configs/vanilla_second_baseline.py \
    --work-dir ./work_dirs/vanilla_second_baseline \
    --auto-scale-lr 2>&1 | tee baseline_training.log

echo ""
echo "===================================================================================="
echo "Baseline training completed. Check baseline_training.log for detailed results."
echo ""
echo "For performance comparison:"
echo "1. Baseline (Standard SECOND): Check baseline_training.log"
echo "2. Adaptive (PhD Research): Check adaptive_fast.log"
echo ""
echo "Key metrics to compare:"
echo "- Training speed (seconds per iteration)"
echo "- Loss convergence pattern"
echo "- Final model performance"
echo "===================================================================================="
