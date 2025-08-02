"""
Adaptive Voxelization Research Configuration (Clean Version)

This config implements learnable adaptive voxelization using the standard
MMDetection3D base config pattern for clarity and maintainability.
"""

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-40e.py',
    '../_base_/default_runtime.py'
]

# Dataset configuration
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Override model components for adaptive voxelization
model = dict(
    type='AdaptiveVoxelNet',  # Custom detector
    
    # 🎓 Research: Learnable Adaptive Voxel Encoder
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        num_features=4,
        min_voxel_size=0.05,
        max_voxel_size=0.50,
        initial_bias=0.2,
        voxel_predictor_hidden=128,
        spatial_encoding_dim=64,
        voxel_aware_hidden=128,
    ),
    
    # 🔬 Research: Multi-Scale Sparse Encoder  
    middle_encoder=dict(
        type='AdaptiveSparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=['conv', 'norm', 'act'],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=[16, 32, 64, 64, 64, 64],
        encoder_paddings=[1, 1, 1, 1, 1, 1],
        block_type='conv_module',
        
        # Multi-scale processing
        num_size_groups=4,
        size_group_ranges=[
            (0.05, 0.15),  # Fine voxels
            (0.15, 0.25),  # Medium-fine voxels
            (0.25, 0.35),  # Medium voxels 
            (0.35, 0.50),  # Coarse voxels
        ],
        fusion_type='attention',
    ),
    
    # Override backbone to match adaptive encoder output
    backbone=dict(
        in_channels=128,  # Match AdaptiveSparseEncoder output
    ),
    
    # Override bbox head for single class
    bbox_head=dict(
        num_classes=1,
    ),
)

# 🎓 Research: Enhanced learning rates for adaptive components
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        lr=0.001, 
        betas=(0.95, 0.99), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'voxel_encoder': dict(lr_mult=1.5),
            'middle_encoder.size_pathways': dict(lr_mult=1.2),
            'middle_encoder.fusion': dict(lr_mult=1.3),
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Working directory
work_dir = './work_dirs/adaptive_multiscale_clean'

print("🚗 Clean Adaptive Voxelization Config Loaded")
print("🔬 Research Features: Learnable voxel sizes + Multi-scale processing")
print("📊 4 parallel pathways with attention-based fusion")
