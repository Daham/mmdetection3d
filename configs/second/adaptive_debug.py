# SIMPLIFIED CONFIG FOR DEBUGGING - NO ADAPTIVE LEARNING
# Use this to test if adaptive processing is causing the delay

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Simplified adaptive bridge - no learning, just basic processing
model = dict(
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        base_voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        min_voxel_size=[0.025, 0.025, 0.05],
        max_voxel_size=[0.2, 0.2, 0.4],
        adaptation_method='learned',
        max_points_per_voxel=32,
        in_channels=4,
        feat_channels=[4],
        learnable_adaptation=False),  # DISABLE LEARNING FOR DEBUGGING
    
    bbox_head=dict(num_classes=1))

# Training settings with more frequent logging for debugging
train_cfg = dict(max_epochs=5, val_interval=1)  # Short test run
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))

# Very frequent logging to see what's happening
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),  # Log every single iteration
    checkpoint=dict(type='CheckpointHook', interval=1)
)
