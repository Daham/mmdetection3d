#!/bin/bash
# Monitor training progress

echo "🔍 Training Monitor - Baseline 01 (Single-Scale)"
echo "================================================"
echo ""

WORK_DIR="work_dirs/baseline_01_full_training"

if [ ! -d "$WORK_DIR" ]; then
    echo "❌ Work directory not found: $WORK_DIR"
    exit 1
fi

# Find latest log file
LOG_FILE=$(ls -t $WORK_DIR/*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "⚠️  No log file found yet. Training may be starting..."
    exit 0
fi

echo "📝 Log file: $LOG_FILE"
echo ""

# Show last 30 lines
echo "📊 Last 30 lines of training log:"
echo "=================================="
tail -30 "$LOG_FILE"

echo ""
echo "=================================="
echo ""

# Check if training is still running
if pgrep -f "train.py.*baseline_01" > /dev/null; then
    echo "✅ Training is RUNNING"
else
    echo "⚠️  Training process not found (may have finished or failed)"
fi

echo ""
echo "💡 Commands:"
echo "  - Watch live: tail -f $LOG_FILE"
echo "  - Check checkpoints: ls -lh $WORK_DIR/*.pth"
echo "  - Monitor GPU: nvidia-smi"
echo ""
