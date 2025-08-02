# ADAPTIVE VOXELIZATION WITH SPARSE CONVOLUTION
# This is the ONLY config you need for adaptive voxel sizes + sparse convolution

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Use SAME voxel sizes as vanilla SECOND for fair comparison
voxel_size = [0.5, 0.5, 0.5]  # Same as vanilla SECOND
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# No custom imports needed - module is registered automatically

# This bridge:
# 1. Learns adaptive voxel sizes during training
# 2. Maps variable voxels to regular grid for sparse convolution
# 3. Handles the compatibility automatically

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,  # Use correct voxel size
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        num_features=4,
        learnable_adaptation=True,         # Enable learning
        adaptation_strength=0.3,          # Conservative strength for stability
        use_attention=False,               # Disable attention initially
        multi_scale=True),                 # Keep multi-scale processing
    
    # Standard sparse convolution works with the bridge output
    bbox_head=dict(num_classes=1))

# Training settings optimized for adaptive voxelization
train_cfg = dict(max_epochs=80, val_interval=10)

# Conservative optimizer settings for stable adaptive training
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        lr=0.0008,              # Slightly lower learning rate for stability
        weight_decay=0.01,
        eps=1e-8                # Numerical stability
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2)  # Gradient clipping for stability
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=80)
]

# Reasonable logging frequency
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),  # Every 50 iterations like standard
    checkpoint=dict(type='CheckpointHook', interval=1)
)
