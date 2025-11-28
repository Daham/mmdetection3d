#!/bin/bash
set -e

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/cyclist_5epochs"
EPOCHS=5
SEED=42

echo ""
echo "========================================================================"
echo "       CYCLIST CLASS COMPARISON (5 epochs each)"
echo "========================================================================"
echo ""
echo "Starting at: $(date)"
echo ""

mkdir -p $WORK_DIR

echo "------------------------------------------------------------------------"
echo "[1/2] Method 1: Single-Scale Cyclist"
echo "------------------------------------------------------------------------"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_05_single_scale_cyclist.py \
    --work-dir ${WORK_DIR}/method1_single \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "Method 1 complete!"
echo ""

echo "------------------------------------------------------------------------"
echo "[2/2] Method 2: Adaptive Multi-Scale Cyclist"
echo "------------------------------------------------------------------------"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_07_adaptive_cyclist.py \
    --work-dir ${WORK_DIR}/method2_adaptive \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "Method 2 complete!"
echo ""

echo "========================================================================"
echo "CYCLIST COMPARISON COMPLETE!"
echo "========================================================================"
echo "Results saved in: ${WORK_DIR}/"
echo "Finished at: $(date)"
echo ""
