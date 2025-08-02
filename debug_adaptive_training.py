#!/usr/bin/env python3
"""
🔍 Adaptive Voxelization Debugging Script
This script helps monitor your adaptive voxelization training
"""

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training_log(log_file='./work_dirs/adaptive_voxel_optimized/vis_data/*.json'):
    """Analyze training logs to understand loss behavior"""
    print("🔍 Analyzing training logs...")
    
    # This is a placeholder - you can extend this to parse actual log files
    print("Monitor these key metrics:")
    print("1. Loss components breakdown:")
    print("   - loss_cls: Should decrease from ~0.74 to <0.5")
    print("   - loss_bbox: Should decrease from ~1.4 to <1.0") 
    print("   - loss_dir: Should stay stable around 0.13")
    print("2. Learning rate: Should vary with scheduler")
    print("3. Gradient norm: Should be stable around 3-5")
    print("4. Memory usage: Should be stable")

def check_model_gradients():
    """Check if gradients are flowing properly"""
    print("\n🔍 Gradient Flow Analysis:")
    print("In your training logs, check:")
    print("1. grad_norm values - should be >0 and <10")
    print("2. If grad_norm is 0 → gradients not flowing")
    print("3. If grad_norm is >50 → exploding gradients")
    
def adaptive_voxel_analysis():
    """Analyze adaptive voxelization behavior"""
    print("\n🔍 Adaptive Voxelization Analysis:")
    print("Key things to monitor:")
    print("1. Voxel count variations during training")
    print("2. Importance threshold effectiveness")
    print("3. Scale range utilization")
    print("4. Memory usage patterns")

def loss_analysis_tips():
    """Provide tips for loss analysis"""
    print("\n📊 Loss Analysis Tips:")
    print("If loss is stuck around 2.3:")
    print("1. Learning rate might be too low → Try 0.005-0.01")
    print("2. Model capacity might be insufficient → Add more channels")
    print("3. Adaptive voxelization might not be helping → Try disabling temporarily")
    print("4. Data augmentation might be too aggressive → Reduce augmentation")
    print("5. Batch size might be too small → Try batch_size=2-4")
    
def quick_fix_suggestions():
    """Provide quick fixes for common issues"""
    print("\n🚀 Quick Fix Suggestions:")
    print("1. INCREASE LEARNING RATE:")
    print("   Change: lr=0.003 → lr=0.01")
    print("\n2. SIMPLIFY ADAPTIVE VOXELIZATION:")
    print("   - importance_threshold: 0.3 → 0.1 (more adaptation)")
    print("   - voxel_size_scale_range: (0.8, 1.5) → (0.9, 1.1) (less variation)")
    print("\n3. ADD LEARNING RATE WARMUP:")
    print("   - Start with lr=0.001 for first 100 iterations")
    print("   - Then ramp up to lr=0.01")
    print("\n4. MONITOR VALIDATION:")
    print("   - Check if validation loss is also stuck")
    print("   - If validation improves but training doesn't → overfitting")

def create_training_script():
    """Create an enhanced training script with debugging"""
    script_content = '''#!/bin/bash
# Enhanced training script with monitoring

echo "🚀 Starting Adaptive Voxelization Training..."
echo "📊 Monitoring: Loss, Gradients, Memory, LR"

# Run training with enhanced logging
python tools/train.py \\
    configs/second/learnable_adaptive_voxel_OPTIMIZED.py \\
    --work-dir ./work_dirs/adaptive_voxel_optimized \\
    --cfg-options \\
        default_hooks.logger.interval=10 \\
        train_cfg.val_interval=1 \\
    2>&1 | tee training_debug.log

echo "✅ Training completed. Check training_debug.log for analysis."
'''
    
    with open('run_adaptive_training.sh', 'w') as f:
        f.write(script_content)
    
    print("\n📝 Created run_adaptive_training.sh")
    print("Run with: chmod +x run_adaptive_training.sh && ./run_adaptive_training.sh")

if __name__ == "__main__":
    print("🔍 ADAPTIVE VOXELIZATION DEBUGGING TOOLKIT")
    print("=" * 50)
    
    analyze_training_log()
    check_model_gradients()
    adaptive_voxel_analysis()
    loss_analysis_tips()
    quick_fix_suggestions()
    create_training_script()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run the optimized config: python tools/train.py configs/second/learnable_adaptive_voxel_OPTIMIZED.py")
    print("2. Monitor the first 100 iterations closely")
    print("3. If loss still stuck, try lr=0.01")
    print("4. Consider temporarily disabling adaptive voxelization to isolate the issue")
