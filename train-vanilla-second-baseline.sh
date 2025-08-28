#!/bin/bash

# PhD Research Training Script - Vanilla SECOND Baseline
# Standard SECOND implementation for comparison
# Author: Daham

echo "🎯 Starting PhD Research Training - Vanilla SECOND Baseline"
echo "📊 Model: Standard SECOND"
echo "🗂️  Dataset: KITTI"
echo "⚡ Features: Original SECOND VFE, standard voxelization"
echo "============================================================"

# Activate virtual environment
source /home/daham/mmdetection_project/mmdet_env/bin/activate

# Clean any previous cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean previous baseline results
if [ -d "./work_dirs/vanilla_second_baseline" ]; then
    echo "Cleaning previous baseline results..."
    rm -rf ./work_dirs/vanilla_second_baseline
fi

# Start training with vanilla SECOND implementation
python tools/train.py configs/second/second_vanilla_baseline_kitti.py

echo "============================================================"
echo "🎓 PhD Research Training Complete - Vanilla SECOND Baseline"
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
