#!/bin/bash
# ==============================================================================
# 3-METHOD COMPARISON: 2 Epochs Each for Quick Validation
# ==============================================================================
# Compares:
#   1. Single-scale HardVFE (standard SECOND)
#   2. Fixed multi-scale (no learning)  
#   3. Learnable multi-scale (your adaptive method)
#
# Usage: ./run_3method_comparison.sh
# Expected runtime: ~30-40 minutes total
# ==============================================================================

set -e  # Exit on error

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/3method_comparison_2epoch"
EPOCHS=2
SEED=42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        3-METHOD COMPARISON (2 epochs each)                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Quickly verify if learnable multi-scale works"
echo "Runtime: ~10 minutes per method = ~30 minutes total"
echo ""
echo "Starting at: $(date)"
echo ""

# Create work directory
mkdir -p $WORK_DIR

# ==============================================================================
# METHOD 1: Single-Scale HardVFE (Baseline)
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔵 [1/3] Method 1: Single-Scale HardVFE (Standard SECOND)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py"
echo "Voxel size: 0.1m (single scale)"
echo ""

$PYTHON tools/train.py \
    configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py \
    --work-dir ${WORK_DIR}/method1_single_scale \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 1 complete!"
echo ""

# ==============================================================================
# METHOD 2: Fixed Multi-Scale (No Learning)
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟡 [2/3] Method 2: Fixed Multi-Scale (Uniform Assignment)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: configs/second/validation_baseline_02_fixed_multiscale_80ep.py"
echo "Voxel scales: [0.05, 0.1, 0.2]m (fixed assignment)"
echo ""

if [ -f "configs/second/validation_baseline_02_fixed_multiscale_80ep.py" ]; then
    $PYTHON tools/train.py \
        configs/second/validation_baseline_02_fixed_multiscale_80ep.py \
        --work-dir ${WORK_DIR}/method2_fixed_multiscale \
        --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED
    echo ""
    echo "✅ Method 2 complete!"
else
    echo "⚠️  Fixed multi-scale config not found, skipping..."
    echo "   This comparison will be Single-Scale vs Learnable only"
fi

echo ""

# ==============================================================================
# METHOD 3: Learnable Multi-Scale (Your Adaptive Method)
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 [3/3] Method 3: Learnable Multi-Scale (Adaptive)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: configs/second/baseline_03_adaptive_simple.py"
echo "Voxel scales: [0.05, 0.1, 0.2]m (LEARNABLE assignment)"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_03_adaptive_simple.py \
    --work-dir ${WORK_DIR}/method3_learnable_multiscale \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 3 complete!"
echo ""

# ==============================================================================
# EXTRACT AND COMPARE RESULTS
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXTRACTING RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to extract Car 3D AP (moderate) from log
extract_result() {
    method_name=$1
    log_dir="${WORK_DIR}/${method_name}"
    
    # Find the log file
    log_file=$(find "$log_dir" -name "*.log" -type f 2>/dev/null | head -1)
    
    if [ -z "$log_file" ] || [ ! -f "$log_file" ]; then
        echo "N/A"
        return
    fi
    
    # Extract the last occurrence of Car_3D_AP11_moderate_strict
    result=$(grep "Car_3D_AP11_moderate_strict" "$log_file" 2>/dev/null | \
             tail -1 | \
             grep -oP 'Car_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
    
    echo "$result"
}

echo "Searching for Car 3D AP@0.70 (moderate difficulty)..."
echo ""

result_1=$(extract_result "method1_single_scale")
result_2=$(extract_result "method2_fixed_multiscale")
result_3=$(extract_result "method3_learnable_multiscale")

# ==============================================================================
# DISPLAY RESULTS TABLE
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    📊 FINAL RESULTS                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌──────────────────────────────────────────┬──────────────┬──────────┐"
echo "│ Method                                   │ 3D AP@0.70   │  Delta   │"
echo "├──────────────────────────────────────────┼──────────────┼──────────┤"

# Method 1 (baseline)
printf "│ %-40s │ %11s%% │ %8s │\n" "Method 1: Single-Scale HardVFE" "$result_1" "baseline"

# Method 2 (if exists)
if [ "$result_2" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_2=$(echo "$result_2 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_2" != "N/A" ]; then
        printf "│ %-40s │ %11s%% │ %+7.2f%% │\n" "Method 2: Fixed Multi-Scale" "$result_2" "$diff_2"
    else
        printf "│ %-40s │ %11s%% │ %8s │\n" "Method 2: Fixed Multi-Scale" "$result_2" "N/A"
    fi
elif [ "$result_2" != "N/A" ]; then
    printf "│ %-40s │ %11s%% │ %8s │\n" "Method 2: Fixed Multi-Scale" "$result_2" "N/A"
fi

# Method 3 (your method)
if [ "$result_3" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_3=$(echo "$result_3 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_3" != "N/A" ]; then
        printf "│ %-40s │ %11s%% │ %+7.2f%% │\n" "Method 3: Learnable Multi-Scale (YOURS)" "$result_3" "$diff_3"
    else
        printf "│ %-40s │ %11s%% │ %8s │\n" "Method 3: Learnable Multi-Scale (YOURS)" "$result_3" "N/A"
    fi
else
    printf "│ %-40s │ %11s%% │ %8s │\n" "Method 3: Learnable Multi-Scale (YOURS)" "$result_3" "N/A"
fi

echo "└──────────────────────────────────────────┴──────────────┴──────────┘"
echo ""

# ==============================================================================
# INTERPRETATION
# ==============================================================================
if [ "$result_3" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_3=$(echo "$result_3 - $result_1" | bc)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 INTERPRETATION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if (( $(echo "$diff_3 > 1.0" | bc -l) )); then
        echo "✅ POSITIVE SIGNAL: Learnable multi-scale shows improvement!"
        echo ""
        echo "   Your adaptive method: ${result_3}%"
        echo "   Single-scale baseline: ${result_1}%"
        echo "   Improvement: +${diff_3}%"
        echo ""
        echo "   📈 This is encouraging! Even at just 2 epochs, you're seeing gains."
        echo "   📈 With full 40-80 epoch training, this gap should widen further."
        echo ""
        echo "   ✅ RECOMMENDED NEXT STEP:"
        echo "      Train for 40 epochs to confirm the trend continues"
        echo ""
    elif (( $(echo "$diff_3 > -0.5" | bc -l) )); then
        echo "⚠️  NEUTRAL: Results are very close (${diff_3}%)"
        echo ""
        echo "   At 2 epochs, it's too early to draw conclusions."
        echo "   The adaptive method needs more epochs to learn optimal scale selection."
        echo ""
        echo "   📊 RECOMMENDED NEXT STEP:"
        echo "      Train for 40 epochs - early results can be misleading"
        echo ""
    else
        echo "❌ CONCERNING: Learnable method underperforming (${diff_3}%)"
        echo ""
        echo "   Even at 2 epochs, we'd expect similar or better performance."
        echo "   This may indicate initialization or hyperparameter issues."
        echo ""
        echo "   🔍 RECOMMENDED NEXT STEPS:"
        echo "      1. Check training logs for NaN, loss spikes, or warnings"
        echo "      2. Review learning rate and Gumbel temperature settings"
        echo "      3. Verify the adaptive mechanism is actually learning"
        echo ""
    fi
else
    echo "⚠️  Could not extract results from logs"
    echo ""
    echo "Manual check:"
    echo "  grep 'Car_3D_AP11_moderate_strict' ${WORK_DIR}/*/20*/*.log"
    echo ""
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 RESULTS SAVED IN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Directory: ${WORK_DIR}/"
echo ""
echo "View detailed logs:"
echo "  tail -100 ${WORK_DIR}/method1_single_scale/20*/*.log"
echo "  tail -100 ${WORK_DIR}/method3_learnable_multiscale/20*/*.log"
echo ""
echo "Finished at: $(date)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 3-METHOD COMPARISON COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
