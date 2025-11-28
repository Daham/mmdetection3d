#!/bin/bash
# Simple 2-epoch test of all 3 baselines

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/quick_2epoch_test"

echo "=========================================="
echo "2-EPOCH QUICK TEST"
echo "=========================================="
echo ""

# Baseline 01: Single-Scale
echo ">>> [1/3] Training Baseline_01 (Single-Scale)..."
$PYTHON tools/train.py \
    configs/second/validation_baseline_01_single_scale_80ep.py \
    --work-dir ${WORK_DIR}/baseline_01 \
    --cfg-options train_cfg.max_epochs=2 randomness.seed=0
echo "✓ Baseline_01 complete"
echo ""

# Baseline 02: Fixed Multi-Scale
echo ">>> [2/3] Training Baseline_02 (Fixed Multi-Scale)..."
$PYTHON tools/train.py \
    configs/second/validation_baseline_02_fixed_multiscale_80ep.py \
    --work-dir ${WORK_DIR}/baseline_02 \
    --cfg-options train_cfg.max_epochs=2 randomness.seed=0
echo "✓ Baseline_02 complete"
echo ""

# Baseline 03: Adaptive
echo ">>> [3/3] Training Baseline_03 (Your Adaptive Method)..."
$PYTHON tools/train.py \
    configs/second/validation_baseline_03_adaptive_80ep.py \
    --work-dir ${WORK_DIR}/baseline_03 \
    --cfg-options train_cfg.max_epochs=2 randomness.seed=0
echo "✓ Baseline_03 complete"
echo ""

echo "=========================================="
echo "ALL DONE! Check results:"
echo "=========================================="
echo ""
echo "grep 'KITTI/Car_3d_moderate' ${WORK_DIR}/*/*/vis_data/scalars.json"
