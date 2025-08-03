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
    voxel_encoder=dict(
        type='MultiScaleVFEWithAttention',
        voxel_scales=[0.05, 0.1, 0.2],
        feature_dim=64,
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=point_cloud_range,
        attention_dim=32,
        scale_embedding_dim=16
    ),
    
    # Enhanced middle encoder for multi-scale features
    middle_encoder=dict(
        type='EnhancedMultiScaleParallelMiddleEncoder',
        in_channels=81,  # 64 features + 16 scale_emb + 1 scale_id
        output_channels=256,
        sparse_shape=[41, 1600, 1408]
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
train_dataloader = dict(batch_size=1)  # Reduce batch size to prevent OOM

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
work_dir = './work_dirs/advanced_multi_scale_attention_v2'
