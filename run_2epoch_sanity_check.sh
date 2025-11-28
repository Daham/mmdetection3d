#!/bin/bash
set -e

echo "⚡⚡⚡ ULTRA-FAST 2-EPOCH SANITY CHECK ⚡⚡⚡"
echo "=========================================="
echo "Purpose: Quick check if Baseline_03 shows ANY improvement"
echo "Time: ~30-40 minutes total"
echo "=========================================="
echo ""

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/2epoch_sanity_check"
EPOCHS=2
SEED=0

mkdir -p $WORK_DIR

echo "Starting at: $(date)"
echo ""

# Function to extract best result
extract_result() {
    log_file=$1
    result=$(grep "KITTI/Car_3d_moderate_strict" "$log_file" 2>/dev/null | \
             grep -oP 'bbox_3d: \K\d+\.\d+' | \
             sort -rn | head -1 || echo "N/A")
    echo "$result"
}

# Baseline 01: Single-Scale
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1/3: Baseline_01 (Single-Scale HardSimpleVFE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON tools/train.py \
    configs/second/validation_baseline_01_single_scale_80ep.py \
    --work-dir ${WORK_DIR}/baseline_01 \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

result_01=$(extract_result "${WORK_DIR}/baseline_01/$(date +%Y%m%d)_*/vis_data/scalars.json")
echo "✓ Baseline_01 Result: ${result_01}%"
echo ""

# Baseline 02: Fixed Multi-Scale
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2/3: Baseline_02 (Fixed Multi-Scale - No Learning)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON tools/train.py \
    configs/second/validation_baseline_02_fixed_multiscale_80ep.py \
    --work-dir ${WORK_DIR}/baseline_02 \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

result_02=$(extract_result "${WORK_DIR}/baseline_02/$(date +%Y%m%d)_*/vis_data/scalars.json")
echo "✓ Baseline_02 Result: ${result_02}%"
echo ""

# Baseline 03: Adaptive (Your Method)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3/3: Baseline_03 (Adaptive Learnable Multi-Scale) 🎯"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON tools/train.py \
    configs/second/validation_baseline_03_adaptive_80ep.py \
    --work-dir ${WORK_DIR}/baseline_03 \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

result_03=$(extract_result "${WORK_DIR}/baseline_03/$(date +%Y%m%d)_*/vis_data/scalars.json")
echo "✓ Baseline_03 Result: ${result_03}%"
echo ""

# Summary
echo "=========================================="
echo "📊 SANITY CHECK RESULTS (2 epochs)"
echo "=========================================="
echo ""
echo "Baseline_01 (Single-Scale):     ${result_01}%"
echo "Baseline_02 (Fixed Multi-Scale): ${result_02}%"
echo "Baseline_03 (Adaptive):          ${result_03}% 🎯"
echo ""

# Parse results and provide interpretation
if [[ "$result_03" != "N/A" && "$result_01" != "N/A" ]]; then
    improvement=$(echo "$result_03 - $result_01" | bc)
    echo "Improvement: ${improvement}%"
    echo ""
    
    # Interpretation
    if (( $(echo "$improvement > 1.0" | bc -l) )); then
        echo "✅ POSITIVE SIGNAL: Baseline_03 shows improvement!"
        echo ""
        echo "📈 What this means:"
        echo "   - Even at 2 epochs, adaptive method is better"
        echo "   - With 40-80 epochs, gap should widen significantly"
        echo "   - Recommended: Run 40-epoch validation for solid proof"
        echo ""
        echo "🎯 Next step:"
        echo "   ./run_fast_validation.sh  # 40 epochs, ~6 hours"
        
    elif (( $(echo "$improvement > -0.5" | bc -l) )); then
        echo "⚠️  NEUTRAL: Results too close to judge"
        echo ""
        echo "📊 What this means:"
        echo "   - 2 epochs too early to see clear differences"
        echo "   - Need more training time to see learning effects"
        echo "   - Adaptive methods need time to learn scale selection"
        echo ""
        echo "🎯 Next step:"
        echo "   ./run_fast_validation.sh  # 40 epochs needed for clarity"
        
    else
        echo "❌ CONCERNING: Baseline_03 underperforming"
        echo ""
        echo "🔍 What this means:"
        echo "   - Method may have initialization issues"
        echo "   - Check training logs for errors/warnings"
        echo "   - May need to adjust hyperparameters"
        echo ""
        echo "🎯 Next step:"
        echo "   1. Check logs: cat ${WORK_DIR}/baseline_03/*/training.log"
        echo "   2. Look for loss spikes, NaN values, or warnings"
        echo "   3. Consider reducing learning rate further"
    fi
else
    echo "⚠️  Could not extract results. Check logs manually:"
    echo "   grep 'KITTI/Car_3d_moderate' ${WORK_DIR}/*/*/training.log"
fi

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "📁 Results saved in: ${WORK_DIR}"
echo ""
echo "🔍 Manual result check:"
echo "   grep -r 'KITTI/Car_3d_moderate' ${WORK_DIR}/*/20*/vis_data/ | grep bbox_3d"
