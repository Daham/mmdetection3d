#!/bin/bash
#
# VALIDATION EXPERIMENT: Compare 3 Baseline Approaches
# 
# Purpose: Confirm learnable multi-scale clearly outperforms:
#   1. Single-scale HardVFE (Baseline_01)
#   2. Fixed multi-scale (Baseline_02)
#
# Total runs: 3 baselines × 3 seeds = 9 training runs
# Estimated time: ~18 hours (2 hours per run × 9)
#

set -e  # Exit on error

# Configuration
PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
TRAIN_SCRIPT="tools/train.py"
NUM_SEEDS=3
BASE_WORK_DIR="work_dirs/validation_experiment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VALIDATION EXPERIMENT${NC}"
echo -e "${BLUE}Comparing 3 Baseline Approaches${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to train a model
train_model() {
    local config=$1
    local seed=$2
    local baseline_name=$3
    local work_dir="${BASE_WORK_DIR}/${baseline_name}_seed${seed}"
    
    echo -e "${GREEN}[$(date +%H:%M:%S)] Training: ${baseline_name} (seed=${seed})${NC}"
    echo -e "${YELLOW}Config: ${config}${NC}"
    echo -e "${YELLOW}Work dir: ${work_dir}${NC}"
    echo ""
    
    $PYTHON $TRAIN_SCRIPT \
        $config \
        --seed $seed \
        --work-dir $work_dir \
        --cfg-options \
            default_hooks.logger.interval=50 \
            default_hooks.checkpoint.interval=10 \
            train_cfg.max_epochs=80 \
        2>&1 | tee ${work_dir}/training.log
    
    # Extract final results
    local val_result=$(grep -A 5 "KITTI/Car_3d" ${work_dir}/training.log | tail -1 || echo "N/A")
    
    echo -e "${GREEN}[$(date +%H:%M:%S)] Completed: ${baseline_name} (seed=${seed})${NC}"
    echo -e "${YELLOW}Result: ${val_result}${NC}"
    echo ""
    echo "=========================================="
    echo ""
}

# Create base work directory
mkdir -p $BASE_WORK_DIR

# ============================================
# BASELINE 01: Single-Scale HardVFE
# ============================================
echo -e "${BLUE}>>> PHASE 1/3: Baseline_01 (Single-Scale HardVFE)${NC}"
echo -e "${YELLOW}Expected: 71-73% 3D AP@0.7 (standard SECOND)${NC}"
echo ""

for seed in $(seq 0 $((NUM_SEEDS-1))); do
    train_model \
        "configs/second/validation_baseline_01_single_scale_80ep.py" \
        $seed \
        "baseline_01_single_scale"
done

# ============================================
# BASELINE 02: Fixed Multi-Scale
# ============================================
echo -e "${BLUE}>>> PHASE 2/3: Baseline_02 (Fixed Multi-Scale)${NC}"
echo -e "${YELLOW}Expected: May underperform Baseline_01 (no adaptive learning)${NC}"
echo ""

for seed in $(seq 0 $((NUM_SEEDS-1))); do
    train_model \
        "configs/second/validation_baseline_02_fixed_multiscale_80ep.py" \
        $seed \
        "baseline_02_fixed_multiscale"
done

# ============================================
# BASELINE 03: Adaptive Learnable Multi-Scale
# ============================================
echo -e "${BLUE}>>> PHASE 3/3: Baseline_03 (Adaptive Learnable)${NC}"
echo -e "${YELLOW}Expected: 74-77% 3D AP@0.7 (SHOULD BE BEST)${NC}"
echo ""

for seed in $(seq 0 $((NUM_SEEDS-1))); do
    train_model \
        "configs/second/validation_baseline_03_adaptive_80ep.py" \
        $seed \
        "baseline_03_adaptive"
done

# ============================================
# SUMMARY AND ANALYSIS
# ============================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL TRAINING COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run analysis script to compute statistics"
echo "2. Generate comparison tables and plots"
echo "3. Verify learnable outperforms others significantly"
echo ""
echo -e "${YELLOW}Run analysis:${NC}"
echo "python tools/analysis_tools/analyze_validation_results.py \\"
echo "    --work-dir ${BASE_WORK_DIR} \\"
echo "    --output validation_results.md"
echo ""
