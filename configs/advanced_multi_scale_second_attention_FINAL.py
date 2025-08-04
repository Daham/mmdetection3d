"""
Advanced Multi-Scale SECOND Configuration with Attention (v2)
============================================================

This configuration uses the base configurations and extends them with our 
advanced multi-scale VFE module with attention mechanisms.

Features:
- Three voxel resolutions (0.05m, 0.1m, 0.2m)
- Separate VFE for each scale
- Attention-based importance weighting
- Learnable scale embeddings
- Enhanced middle encoder

Author: PhD Research Implementation
Date: August 3, 2025
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-car.py',
    './_base_/schedules/cyclic-2e.py',
    './_base_/default_runtime.py'
]

# Data root configuration
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Point cloud range and voxel settings
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration with our advanced multi-scale components
model = dict(
    # Keep VoxelNet type from base config
    type='VoxelNet',
    
    # Update data preprocessor for our needs
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],  # Base voxel size
            max_voxels=(12000, 30000)
        )
    ),
    
    # Our advanced multi-scale VFE
    # 🎓 YOUR PhD RESEARCH VFE - ENHANCED 10-Scale Adaptive Voxelization
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',  # Your research contribution!
        # 🚀 ENHANCED: 10 scales with automatic optimal generation (0.01m to 1.0m)
        num_scales=10,                        # 10-scale multi-resolution voxelization
        scale_net_hidden_dims=[64, 32],       # ScaleNet architecture for 10-scale selection
        gumbel_temperature=5.0,               # Differentiable scale selection with diversity
        vfe_channels=[32, 64],
        fusion_channels=128,
        output_channels=64,                   # Matches middle encoder input
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=point_cloud_range
        # Note: voxel_scales auto-generated as [0.010m, 0.017m, 0.028m, 0.046m, 0.077m, 
        #       0.129m, 0.215m, 0.359m, 0.599m, 1.000m] when num_scales=10
    ),
    
    # ✅ CUDA-SAFE middle encoder (PhD research preserved in VFE above)
    middle_encoder=dict(
        type='SparseEncoder',  # Standard CUDA-compatible encoder
        in_channels=65,        # Match your research VFE output (64 + 1 scale info)
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=256,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    # ✅ FIXED: Update backbone input channels to match neck output (512 = 256 + 256)
    backbone=dict(
        type='SECOND',
        in_channels=512,  # CRITICAL: Match neck output [256, 256] concatenated
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    
    # Override bbox head for single class
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            ranges=[[0, -40, -0.6, 70.4, 40, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False
        )
    ),
    
    # Single train configuration for single anchor scale
    train_cfg=dict(
        _delete_=True,  # Delete base config assigners list
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.35,
            neg_iou_thr=0.2,
            min_pos_iou=0.2,
            ignore_iof_thr=-1
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    )
)

# Training configuration
# Training configuration
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=5)

# ✅ FIX: Proper dataloader configuration - set num_workers > 0 for persistent_workers
train_dataloader = dict(
    batch_size=1,  # Reduced batch size for memory efficiency
    num_workers=2,  # CRITICAL: Must be > 0 when persistent_workers=True
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,  # CRITICAL: Must be > 0 when persistent_workers=True
    persistent_workers=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,  # CRITICAL: Must be > 0 when persistent_workers=True
    persistent_workers=True
)

# Optimizer configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        eta_min=1e-6,
        begin=0,
        end=40,
        by_epoch=True
    )
]

# Work directory
work_dir = './work_dirs/backbone_channel_fix'
