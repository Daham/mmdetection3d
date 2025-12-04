#!/bin/bash
# ==============================================================================
# PEDESTRIAN - METHOD 3 ONLY (Adaptive/Learnable Multi-Scale)
# ==============================================================================

set -e

PYTHON="/home/daham/mmdetection_project/mmdet_env/bin/python"
WORK_DIR="work_dirs/pedestrian_method3_5epochs"
EPOCHS=5
SEED=42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  PEDESTRIAN - METHOD 3: Adaptive Multi-Scale (5 epochs)       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting at: $(date)"
echo ""

mkdir -p $WORK_DIR

# ==============================================================================
# METHOD 3: Adaptive Multi-Scale Pedestrian
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 Method 3: Adaptive Multi-Scale Pedestrian (Learnable)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON tools/train.py \
    configs/second/baseline_06_adaptive_pedestrian.py \
    --work-dir ${WORK_DIR} \
    --cfg-options train_cfg.max_epochs=$EPOCHS randomness.seed=$SEED

echo ""
echo "✅ Training complete!"
echo ""

# ==============================================================================
# EXTRACT RESULTS FROM LOG
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXTRACTING RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

LOG_FILE=$(find "${WORK_DIR}" -name "*.log" -type f 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ Error: Log file not found!"
    exit 1
fi

echo "Log file: $LOG_FILE"
echo ""

# Extract all 5 epochs of results for Pedestrian class
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         PEDESTRIAN 3D AP RESULTS - ALL 5 EPOCHS                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌───────┬──────────┬────────────┬──────────┐"
echo "│ Epoch │   Easy   │  Moderate  │   Hard   │"
echo "├───────┼──────────┼────────────┼──────────┤"

for epoch in {1..5}; do
    # Search for validation results for this epoch
    easy=$(grep "Epoch(val) \[$epoch\]" "$LOG_FILE" | \
           grep -oP 'Pedestrian_3D_AP40_easy_strict: \K[\d.]+' | tail -1)
    
    moderate=$(grep "Epoch(val) \[$epoch\]" "$LOG_FILE" | \
               grep -oP 'Pedestrian_3D_AP40_moderate_strict: \K[\d.]+' | tail -1)
    
    hard=$(grep "Epoch(val) \[$epoch\]" "$LOG_FILE" | \
           grep -oP 'Pedestrian_3D_AP40_hard_strict: \K[\d.]+' | tail -1)
    
    # Default to N/A if not found
    easy=${easy:-"N/A"}
    moderate=${moderate:-"N/A"}
    hard=${hard:-"N/A"}
    
    printf "│   %d   │ %8s │  %8s  │ %8s │\n" "$epoch" "$easy" "$moderate" "$hard"
done

echo "└───────┴──────────┴────────────┴──────────┘"
echo ""

# Also show final epoch 5 results more prominently
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 FINAL RESULTS (EPOCH 5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

final_easy=$(grep "Epoch(val) \[5\]" "$LOG_FILE" | \
             grep -oP 'Pedestrian_3D_AP40_easy_strict: \K[\d.]+' | tail -1)

final_moderate=$(grep "Epoch(val) \[5\]" "$LOG_FILE" | \
                 grep -oP 'Pedestrian_3D_AP40_moderate_strict: \K[\d.]+' | tail -1)

final_hard=$(grep "Epoch(val) \[5\]" "$LOG_FILE" | \
             grep -oP 'Pedestrian_3D_AP40_hard_strict: \K[\d.]+' | tail -1)

echo "Easy:     ${final_easy:-N/A}%"
echo "Moderate: ${final_moderate:-N/A}%"
echo "Hard:     ${final_hard:-N/A}%"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PEDESTRIAN METHOD 3 COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved in: ${WORK_DIR}/"
echo "Checkpoints: ${WORK_DIR}/epoch_{1..5}.pth"
echo "Finished at: $(date)"
echo ""
