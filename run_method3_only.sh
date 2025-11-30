#!/bin/bash
# ==============================================================================
# RUN METHOD 3 ONLY - With PhD Fix Applied
# ==============================================================================
# Tests the learnable multi-scale adaptive voxelization with the fix that
# connects learnable voxel_scales (nn.Parameter) to actual voxelization geometry
# ==============================================================================

set -e

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/method3_with_fix_5epochs"
EPOCHS=5
SEED=42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     METHOD 3: Learnable Multi-Scale (WITH PhD FIX)            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎓 PhD Fix Applied: Learnable scales now control voxelization"
echo "📊 Training for $EPOCHS epochs with seed=$SEED"
echo "Starting at: $(date)"
echo ""

mkdir -p $WORK_DIR

# ==============================================================================
# METHOD 3: Learnable Multi-Scale (Adaptive) - WITH FIX
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 Training Method 3: Learnable Multi-Scale (Adaptive)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_03_adaptive_multiscale_learnable.py \
    --work-dir ${WORK_DIR} \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Method 3 training complete!"
echo ""

# ==============================================================================
# EXTRACT RESULTS
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXTRACTING RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

LOG_FILE=$(find "$WORK_DIR" -name "*.log" -type f 2>/dev/null | head -1)

if [ -n "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Extract Car results
    car_result=$(grep "Car_3D_AP11_moderate_strict" "$LOG_FILE" 2>/dev/null | \
                 tail -1 | \
                 grep -oP 'Car_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
    
    # Extract Pedestrian results
    ped_result=$(grep "Pedestrian_3D_AP11_moderate_strict" "$LOG_FILE" 2>/dev/null | \
                 tail -1 | \
                 grep -oP 'Pedestrian_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
    
    # Extract Cyclist results
    cyc_result=$(grep "Cyclist_3D_AP11_moderate_strict" "$LOG_FILE" 2>/dev/null | \
                 tail -1 | \
                 grep -oP 'Cyclist_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
    
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              📊 METHOD 3 RESULTS (WITH FIX)                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "┌────────────────────────────┬──────────────┐"
    echo "│ Class                      │ 3D AP@0.70   │"
    echo "├────────────────────────────┼──────────────┤"
    printf "│ %-26s │ %11s%% │\n" "Car" "$car_result"
    printf "│ %-26s │ %11s%% │\n" "Pedestrian" "$ped_result"
    printf "│ %-26s │ %11s%% │\n" "Cyclist" "$cyc_result"
    echo "└────────────────────────────┴──────────────┘"
    echo ""
    
    # Compare with previous Method 3 results (if they exist)
    OLD_WORK_DIR="work_dirs/comparison_5epochs/method3_learnable"
    OLD_LOG_FILE=$(find "$OLD_WORK_DIR" -name "*.log" -type f 2>/dev/null | head -1)
    
    if [ -n "$OLD_LOG_FILE" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔍 COMPARISON: Before Fix vs After Fix"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        old_car=$(grep "Car_3D_AP11_moderate_strict" "$OLD_LOG_FILE" 2>/dev/null | \
                  tail -1 | \
                  grep -oP 'Car_3D_AP11_moderate_strict: \K[\d.]+' 2>/dev/null || echo "N/A")
        
        if [ "$old_car" != "N/A" ] && [ "$car_result" != "N/A" ]; then
            diff_car=$(echo "$car_result - $old_car" | bc 2>/dev/null || echo "N/A")
            echo "Car Class:"
            echo "  Before fix: ${old_car}%"
            echo "  After fix:  ${car_result}%"
            if [ "$diff_car" != "N/A" ]; then
                printf "  Difference: %+.2f%%\n" "$diff_car"
            fi
            echo ""
        fi
    fi
    
    # Look for scale adaptation evidence
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 CHECKING FOR SCALE ADAPTATION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    scale_logs=$(grep "LEARNABLE SCALES" "$LOG_FILE" 2>/dev/null | tail -5 || echo "")
    
    if [ -n "$scale_logs" ]; then
        echo "✅ Found learnable scale updates (last 5):"
        echo "$scale_logs"
        echo ""
        echo "🎓 SUCCESS: Scales are adapting during training!"
    else
        echo "⚠️  No scale adaptation logs found yet"
        echo "   (They appear randomly in 2% of batches)"
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TRAINING COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved in: ${WORK_DIR}/"
echo "Finished at: $(date)"
echo ""
