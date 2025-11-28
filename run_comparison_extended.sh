#!/bin/bash
# ==============================================================================
# 4-METHOD COMPARISON: 5 Epochs Each (with PointPillars)
# ==============================================================================
# Compares:
#   0. PointPillars (reference baseline)
#   1. Single-scale HardVFE (SECOND baseline)
#   2. Fixed multi-scale (no learning)
#   3. Learnable multi-scale (adaptive - OURS)
# ==============================================================================

set -e

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/comparison_5epochs_with_references"
EPOCHS=5
SEED=42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     4-METHOD COMPARISON (5 epochs each) + PointPillars         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting at: $(date)"
echo ""

mkdir -p $WORK_DIR

# ==============================================================================
# METHOD 0: PointPillars (Reference)
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔶 [0/4] Method 0: PointPillars (Reference)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_00_pointpillars.py \
    --work-dir ${WORK_DIR}/method0_pointpillars \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 0 complete!"
echo ""

# ==============================================================================
# METHOD 1: Single-Scale HardVFE
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔵 [1/4] Method 1: Single-Scale HardVFE (SECOND)"
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
echo "🟡 [2/4] Method 2: Fixed Multi-Scale (No Learning)"
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
# METHOD 3: Learnable Multi-Scale (Adaptive - OURS)
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 [3/4] Method 3: Learnable Multi-Scale (Adaptive - OURS)"
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

result_0=$(extract_result "method0_pointpillars")
result_1=$(extract_result "method1_single")
result_2=$(extract_result "method2_fixed")
result_3=$(extract_result "method3_learnable")

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                      📊 FINAL RESULTS                             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌────────────────────────────────────────────────┬──────────────┬──────────┐"
echo "│ Method                                         │ 3D AP@0.70   │  Delta   │"
echo "├────────────────────────────────────────────────┼──────────────┼──────────┤"

# Use Single-Scale as baseline
printf "│ %-46s │ %11s%% │ %8s │\n" "Method 1: Single-Scale HardVFE (baseline)" "$result_1" "baseline"

if [ "$result_0" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_0=$(echo "$result_0 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_0" != "N/A" ]; then
        printf "│ %-46s │ %11s%% │ %+7.2f%% │\n" "Method 0: PointPillars (reference)" "$result_0" "$diff_0"
    fi
fi

if [ "$result_2" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_2=$(echo "$result_2 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_2" != "N/A" ]; then
        printf "│ %-46s │ %11s%% │ %+7.2f%% │\n" "Method 2: Fixed Multi-Scale" "$result_2" "$diff_2"
    fi
fi

if [ "$result_3" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_3=$(echo "$result_3 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_3" != "N/A" ]; then
        printf "│ %-46s │ %11s%% │ %+7.2f%% │\n" "Method 3: Learnable Multi-Scale (OURS)" "$result_3" "$diff_3"
    fi
fi

echo "└────────────────────────────────────────────────┴──────────────┴──────────┘"
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
        echo "   → Our method outperforms both baselines"
        if [ "$result_0" != "N/A" ]; then
            diff_vs_pp=$(echo "$result_3 - $result_0" | bc)
            if (( $(echo "$diff_vs_pp > 0" | bc -l) )); then
                echo "   → Also outperforms PointPillars (+${diff_vs_pp}%)"
            else
                echo "   → Still behind PointPillars (${diff_vs_pp}%) but improving"
            fi
        fi
        echo "   → Train for 40-80 epochs for publication"
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
