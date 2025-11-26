#!/bin/bash
# ==============================================================================
# 3-METHOD COMPARISON: 2 Epochs Each
# ==============================================================================
# Compares:
#   1. Single-scale HardVFE
#   2. Fixed multi-scale (no learning)
#   3. Learnable multi-scale (adaptive)
# ==============================================================================

set -e

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/comparison_5epochs"
EPOCHS=5
SEED=42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        3-METHOD COMPARISON (5 epochs each)                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting at: $(date)"
echo ""

mkdir -p $WORK_DIR

# ==============================================================================
# METHOD 1: Single-Scale HardVFE
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔵 [1/3] Method 1: Single-Scale HardVFE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_01_single_scale_hardvfe.py \
    --work-dir ${WORK_DIR}/method1_single \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 1 complete!"
echo ""

# ==============================================================================
# METHOD 2: Fixed Multi-Scale
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟡 [2/3] Method 2: Fixed Multi-Scale"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_02_fixed_multiscale_gumbel.py \
    --work-dir ${WORK_DIR}/method2_fixed \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 2 complete!"
echo ""

# ==============================================================================
# METHOD 3: Learnable Multi-Scale (Adaptive)
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 [3/3] Method 3: Learnable Multi-Scale (Adaptive)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_03_adaptive_multiscale_learnable.py \
    --work-dir ${WORK_DIR}/method3_learnable \
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

extract_result() {
    method_name=$1
    log_dir="${WORK_DIR}/${method_name}"
    log_file=$(find "$log_dir" -name "*.log" -type f 2>/dev/null | head -1)
    
    if [ -z "$log_file" ]; then
        echo "N/A"
        return
    fi
    
    result=$(grep "Car_3D_AP11_moderate_strict" "$log_file" 2>/dev/null | \
             tail -1 | \
             grep -oP 'Car_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
    
    echo "$result"
}

result_1=$(extract_result "method1_single")
result_2=$(extract_result "method2_fixed")
result_3=$(extract_result "method3_learnable")

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    📊 FINAL RESULTS                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌──────────────────────────────────────────┬──────────────┬──────────┐"
echo "│ Method                                   │ 3D AP@0.70   │  Delta   │"
echo "├──────────────────────────────────────────┼──────────────┼──────────┤"

printf "│ %-40s │ %11s%% │ %8s │\n" "Method 1: Single-Scale HardVFE" "$result_1" "baseline"

if [ "$result_2" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_2=$(echo "$result_2 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_2" != "N/A" ]; then
        printf "│ %-40s │ %11s%% │ %+7.2f%% │\n" "Method 2: Fixed Multi-Scale" "$result_2" "$diff_2"
    fi
fi

if [ "$result_3" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_3=$(echo "$result_3 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_3" != "N/A" ]; then
        printf "│ %-40s │ %11s%% │ %+7.2f%% │\n" "Method 3: Learnable Multi-Scale" "$result_3" "$diff_3"
    fi
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
        echo "✅ POSITIVE: Learnable shows improvement (+${diff_3}%)"
        echo "   → Train for 40-80 epochs for paper"
    elif (( $(echo "$diff_3 > -0.5" | bc -l) )); then
        echo "⚠️  NEUTRAL: Results are close (${diff_3}%)"
        echo "   → Train longer to see learning effects"
    else
        echo "❌ CONCERNING: Underperforming (${diff_3}%)"
        echo "   → Check logs for issues"
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ COMPARISON COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved in: ${WORK_DIR}/"
echo "Finished at: $(date)"
echo ""
