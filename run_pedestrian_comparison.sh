#!/bin/bash
# ==============================================================================
# PEDESTRIAN CLASS COMPARISON: 2 Methods, 5 Epochs Each
# ==============================================================================
# Compares:
#   1. Single-scale baseline
#   2. Learnable multi-scale (adaptive)
# ==============================================================================

set -e

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/pedestrian_5epochs"
EPOCHS=5
SEED=42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     PEDESTRIAN CLASS COMPARISON (5 epochs each)                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting at: $(date)"
echo ""

mkdir -p $WORK_DIR

# ==============================================================================
# METHOD 1: Single-Scale Pedestrian
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔵 [1/2] Method 1: Single-Scale Pedestrian"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_04_single_scale_pedestrian.py \
    --work-dir ${WORK_DIR}/method1_single \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 1 complete!"
echo ""

# ==============================================================================
# METHOD 2: Adaptive Multi-Scale Pedestrian
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 [2/2] Method 2: Adaptive Multi-Scale Pedestrian"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_06_adaptive_pedestrian.py \
    --work-dir ${WORK_DIR}/method2_adaptive \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 2 complete!"
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
    
    result=$(grep "Pedestrian_3D_AP11_moderate_strict" "$log_file" 2>/dev/null | \
             tail -1 | \
             grep -oP 'Pedestrian_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
    
    echo "$result"
}

result_1=$(extract_result "method1_single")
result_2=$(extract_result "method2_adaptive")

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              📊 PEDESTRIAN CLASS RESULTS                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌──────────────────────────────────────────┬──────────────┬──────────┐"
echo "│ Method                                   │ 3D AP@0.50   │  Delta   │"
echo "├──────────────────────────────────────────┼──────────────┼──────────┤"

printf "│ %-40s │ %11s%% │ %8s │\n" "Method 1: Single-Scale" "$result_1" "baseline"

if [ "$result_2" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_2=$(echo "$result_2 - $result_1" | bc 2>/dev/null || echo "N/A")
    if [ "$diff_2" != "N/A" ]; then
        printf "│ %-40s │ %11s%% │ %+7.2f%% │\n" "Method 2: Adaptive Multi-Scale" "$result_2" "$diff_2"
    fi
fi

echo "└──────────────────────────────────────────┴──────────────┴──────────┘"
echo ""

# ==============================================================================
# INTERPRETATION
# ==============================================================================
if [ "$result_2" != "N/A" ] && [ "$result_1" != "N/A" ]; then
    diff_2=$(echo "$result_2 - $result_1" | bc)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 INTERPRETATION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if (( $(echo "$diff_2 > 1.0" | bc -l) )); then
        echo "✅ POSITIVE: Adaptive shows improvement (+${diff_2}%)"
        echo "   → Pedestrian class benefits from adaptive voxelization"
    elif (( $(echo "$diff_2 > -0.5" | bc -l) )); then
        echo "⚠️  NEUTRAL: Results are close (${diff_2}%)"
        echo "   → Train longer to see learning effects"
    else
        echo "❌ CONCERNING: Underperforming (${diff_2}%)"
        echo "   → Check logs for issues"
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PEDESTRIAN COMPARISON COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved in: ${WORK_DIR}/"
echo "Finished at: $(date)"
echo ""
