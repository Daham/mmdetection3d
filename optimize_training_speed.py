#!/usr/bin/env python3
"""
Quick fix script to optimize training speed
"""

import os
import shutil

def optimize_config():
    """Create optimized config for faster training"""
    
    # Find existing config
    config_paths = [
        'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py',
        'configs/second/adaptive_voxel_second_secfpn_6x8_80e_kitti_3d_car.py'
    ]
    
    base_config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            base_config = config_path
            break
    
    if not base_config:
        print("❌ No suitable config found")
        return
    
    print(f"📝 Using base config: {base_config}")
    
    # Create optimized config
    optimized_config = """
# Fast training config - optimized for speed
_base_ = ['../second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py']

# Optimize data loading
train_dataloader = dict(
    batch_size=8,  # Increased from 6
    num_workers=8,  # More workers
    persistent_workers=True,  # Keep workers alive
    pin_memory=True  # Faster GPU transfer
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True
)

# Enable performance optimizations
env_cfg = dict(
    cudnn_benchmark=True,  # Speed up convolutions
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

# Mixed precision training for speed
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Faster training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,  # Reduced from 80
    val_interval=5   # Validate less frequently
)

# Reduce model complexity slightly
model = dict(
    pts_voxel_encoder=dict(
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=[0.1, 0.1, 0.2]  # Slightly larger voxels
    )
)

# Log less frequently
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),  # Log every 20 iters instead of 10
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2)
)
"""
    
    # Write optimized config
    fast_config_path = 'configs/second/fast_second_kitti.py'
    with open(fast_config_path, 'w') as f:
        f.write(optimized_config)
    
    print(f"✅ Created fast config: {fast_config_path}")
    return fast_config_path

def create_quick_test_script():
    """Create a script for quick performance testing"""
    
    test_script = """#!/bin/bash
# Quick training test - 50 iterations only

echo "🚀 Starting quick performance test..."

cd /home/daham/mmdetection_project/mmdetection3d

# Run with time measurement
time /home/daham/mmdetection_project/mmdet_env/bin/python tools/train.py \\
    configs/second/fast_second_kitti.py \\
    --work-dir work_dirs/speed_test \\
    --cfg-options train_cfg.max_iters=50 \\
    2>&1 | tee speed_test.log

echo "🎯 Test complete! Check speed_test.log for timing results"

# Extract timing info
echo "📊 Performance Summary:"
grep "time:" speed_test.log | tail -5
"""
    
    with open('quick_speed_test.sh', 'w') as f:
        f.write(test_script)
    
    os.chmod('quick_speed_test.sh', 0o755)
    print("✅ Created quick_speed_test.sh")

if __name__ == "__main__":
    print("🔧 TRAINING SPEED OPTIMIZER")
    print("=" * 40)
    
    # Create optimized config
    fast_config = optimize_config()
    
    # Create test script
    create_quick_test_script()
    
    print("\n" + "=" * 40)
    print("🎯 OPTIMIZATION COMPLETE")
    print("\n📋 Next steps:")
    print("1. Run: ./quick_speed_test.sh")
    print("2. Monitor: watch -n 1 nvidia-smi")
    print("3. Check logs for improved timing")
    print("\n💡 Expected improvement: 3.05s → <1.5s per iteration")
