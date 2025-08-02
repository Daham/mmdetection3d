#!/bin/bash

# Quick diagnosis script to compare vanilla vs adaptive behavior
# Run this on your training machine

echo "🔍 Quick Diagnosis: Vanilla vs Adaptive VFE Comparison"
echo "============================================================"

# Test 1: True vanilla SECOND (should work perfectly)
echo "🧪 Test 1: True Vanilla SECOND (HardSimpleVFE)"
echo "Expected: Loss should decrease from ~2.8 to ~2.0 in first 100 iterations"
python tools/train.py configs/second/test_true_vanilla.py \
    --work-dir ./work_dirs/test_vanilla \
    --cfg-options train_cfg.max_epochs=1 \
    > vanilla_test.log 2>&1 &

VANILLA_PID=$!

# Test 2: Our adaptive module (currently identical to vanilla)
echo "🧪 Test 2: Adaptive Module (currently disabled - should be identical)"
echo "Expected: Exact same behavior as Test 1"
python tools/train.py configs/second/debug_vanilla_adaptive.py \
    --work-dir ./work_dirs/test_adaptive_debug \
    --cfg-options train_cfg.max_epochs=1 \
    > adaptive_test.log 2>&1 &

ADAPTIVE_PID=$!

# Wait a bit and show first results
sleep 30

echo ""
echo "📊 First 30 seconds results:"
echo ""
echo "🟢 True Vanilla Results:"
tail -n 5 vanilla_test.log | grep -E "(loss:|grad_norm:|INFO)"

echo ""
echo "🔶 Adaptive Results:"
tail -n 5 adaptive_test.log | grep -E "(loss:|grad_norm:|INFO)"

echo ""
echo "⏱️  Tests running in background (PIDs: $VANILLA_PID, $ADAPTIVE_PID)"
echo "📋 Monitor with:"
echo "   tail -f vanilla_test.log"
echo "   tail -f adaptive_test.log" 
echo "🛑 Stop with:"
echo "   kill $VANILLA_PID $ADAPTIVE_PID"

# Show comparison after 2 minutes
sleep 90

echo ""
echo "📊 After 2 minutes:"
echo ""
echo "🟢 True Vanilla (last 3 lines):"
tail -n 3 vanilla_test.log | grep -E "(loss:|grad_norm:|INFO)"

echo ""
echo "🔶 Adaptive (last 3 lines):"
tail -n 3 adaptive_test.log | grep -E "(loss:|grad_norm:|INFO)"

echo ""
echo "🎯 Analysis:"
if grep -q "AdaptiveSparseBridge forward called" adaptive_test.log; then
    echo "✅ Our adaptive module IS being used"
else
    echo "❌ Our adaptive module is NOT being called - config issue!"
fi

echo ""
echo "📈 Compare the loss curves:"
echo "   - If both show similar loss reduction: Our module works correctly"
echo "   - If adaptive plateaus: There's still an issue in our implementation"
echo "   - If adaptive doesn't call our module: Registration/config problem"
