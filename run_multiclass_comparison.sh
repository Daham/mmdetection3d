#!/bin/bash

##############################################################################
# Multi-Class Comparison Script: Pedestrian & Cyclist
# 
# Compares single-scale vs adaptive multi-scale voxelization for:
# - Pedestrian class
# - Cyclist class
#
# All methods trained for 5 epochs with seed=42 for fair comparison
##############################################################################

set -e  # Exit on error

# Configuration
EPOCHS=5
SEED=42
PYTHON=python
TOOLS_DIR=tools

echo "========================================================================"
echo "Multi-Class Comparison: Pedestrian & Cyclist"
echo "========================================================================"
echo "Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Random Seed: $SEED"
echo "  - Classes: Pedestrian, Cyclist"
echo "  - Methods per class: Single-Scale, Adaptive Multi-Scale"
echo "========================================================================"
echo ""

##############################################################################
# PEDESTRIAN CLASS
##############################################################################

echo "========================================================================"
echo "PEDESTRIAN CLASS EXPERIMENTS"
echo "========================================================================"

echo ""
echo "------------------------------------------------------------------------"
echo "METHOD 1: Single-Scale Pedestrian (Baseline)"
echo "------------------------------------------------------------------------"
$PYTHON $TOOLS_DIR/train.py \
    configs/second/baseline_04_single_scale_pedestrian.py \
    --work-dir work_dirs/multiclass_5epochs/pedestrian_single \
    --seed $SEED \
    --cfg-options train_cfg.max_epochs=$EPOCHS

echo ""
echo "------------------------------------------------------------------------"
echo "METHOD 2: Adaptive Multi-Scale Pedestrian"
echo "------------------------------------------------------------------------"
$PYTHON $TOOLS_DIR/train.py \
    configs/second/baseline_06_adaptive_pedestrian.py \
    --work-dir work_dirs/multiclass_5epochs/pedestrian_adaptive \
    --seed $SEED \
    --cfg-options train_cfg.max_epochs=$EPOCHS

##############################################################################
# CYCLIST CLASS
##############################################################################

echo ""
echo "========================================================================"
echo "CYCLIST CLASS EXPERIMENTS"
echo "========================================================================"

echo ""
echo "------------------------------------------------------------------------"
echo "METHOD 3: Single-Scale Cyclist (Baseline)"
echo "------------------------------------------------------------------------"
$PYTHON $TOOLS_DIR/train.py \
    configs/second/baseline_05_single_scale_cyclist.py \
    --work-dir work_dirs/multiclass_5epochs/cyclist_single \
    --seed $SEED \
    --cfg-options train_cfg.max_epochs=$EPOCHS

echo ""
echo "------------------------------------------------------------------------"
echo "METHOD 4: Adaptive Multi-Scale Cyclist"
echo "------------------------------------------------------------------------"
$PYTHON $TOOLS_DIR/train.py \
    configs/second/baseline_07_adaptive_cyclist.py \
    --work-dir work_dirs/multiclass_5epochs/cyclist_adaptive \
    --seed $SEED \
    --cfg-options train_cfg.max_epochs=$EPOCHS

##############################################################################
# RESULTS EXTRACTION
##############################################################################

echo ""
echo "========================================================================"
echo "EXTRACTING RESULTS"
echo "========================================================================"

echo ""
echo "=== PEDESTRIAN CLASS RESULTS ==="
echo ""
echo "Single-Scale Pedestrian:"
find work_dirs/multiclass_5epochs/pedestrian_single -name "*.log" -type f -exec tail -20 {} \; | grep "Pedestrian_3D_AP11_moderate_strict" | tail -1

echo ""
echo "Adaptive Pedestrian:"
find work_dirs/multiclass_5epochs/pedestrian_adaptive -name "*.log" -type f -exec tail -20 {} \; | grep "Pedestrian_3D_AP11_moderate_strict" | tail -1

echo ""
echo "=== CYCLIST CLASS RESULTS ==="
echo ""
echo "Single-Scale Cyclist:"
find work_dirs/multiclass_5epochs/cyclist_single -name "*.log" -type f -exec tail -20 {} \; | grep "Cyclist_3D_AP11_moderate_strict" | tail -1

echo ""
echo "Adaptive Cyclist:"
find work_dirs/multiclass_5epochs/cyclist_adaptive -name "*.log" -type f -exec tail -20 {} \; | grep "Cyclist_3D_AP11_moderate_strict" | tail -1

echo ""
echo "========================================================================"
echo "COMPARISON COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved in:"
echo "  - work_dirs/multiclass_5epochs/pedestrian_single/"
echo "  - work_dirs/multiclass_5epochs/pedestrian_adaptive/"
echo "  - work_dirs/multiclass_5epochs/cyclist_single/"
echo "  - work_dirs/multiclass_5epochs/cyclist_adaptive/"
echo ""
echo "To extract detailed metrics, check the log files in each directory."
echo "========================================================================"
