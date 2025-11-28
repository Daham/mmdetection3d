#!/bin/bash
# Dead simple 2-epoch test - just modify epochs on working configs

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/simple_2epoch_test"

echo "=========================================="
echo "SIMPLE 2-EPOCH TEST"
echo "=========================================="
echo ""

# Test 1: Standard SECOND (working baseline)
echo ">>> [1/2] Baseline_01: Standard SECOND..."
$PYTHON tools/train.py \
    configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py \
    --work-dir ${WORK_DIR}/baseline_01_standard \
    --cfg-options train_cfg.max_epochs=2 randomness.seed=0

echo ""
echo "✓ Baseline_01 done"
echo ""

# Test 2: Your adaptive method  
echo ">>> [2/2] Baseline_03: Your Adaptive Method..."
$PYTHON tools/train.py \
    configs/second/validation_baseline_03_adaptive_80ep.py \
    --work-dir ${WORK_DIR}/baseline_03_adaptive \
    --cfg-options train_cfg.max_epochs=2 train_cfg.by_epoch=True randomness.seed=0

echo ""
echo "✓ Baseline_03 done"
echo ""

echo "=========================================="
echo "RESULTS:"
echo "=========================================="
echo ""
echo "Baseline_01 (Standard SECOND):"
tail -20 ${WORK_DIR}/baseline_01_standard/*/vis_data/scalars.json | grep "Car_3d_moderate" || echo "Check log manually"
echo ""
echo "Baseline_03 (Your Adaptive):"
tail -20 ${WORK_DIR}/baseline_03_adaptive/*/vis_data/scalars.json | grep "Car_3d_moderate" || echo "Check log manually"
