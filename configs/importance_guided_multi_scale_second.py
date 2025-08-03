"""
Importance-Guided Multi-Scale SECOND Configuration
=================================================

This configuration implements the memory-efficient importance-guided multi-scale VFE
following the architecture:

Point Cloud → Importance Net → Top-K Selection → Multi-Scale Voxelization → Lightweight VFE

Features:
- Lightweight point importance network (3-layer MLP)
- Top-K point selection (70% most important points)
- Multi-scale voxelization on filtered points
- Shared lightweight VFE with scale embeddings
- Attention-based feature fusion
- Memory-efficient design

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

# Model configuration with importance-guided multi-scale components
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
    
    # 🌟 Importance-guided multi-scale VFE
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        voxel_scales=[0.05, 0.1, 0.2],  # Multi-scale voxelization
        feature_dim=64,
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=point_cloud_range,
        
        # Importance network configuration
        importance_keep_ratio=0.7,  # Keep 70% of most important points
        importance_hidden_dims=[64, 32, 16],  # 3-layer lightweight MLP
        importance_dropout=0.1,
        
        # Lightweight VFE configuration
        vfe_channels=[32, 64],  # Smaller channels for efficiency
        scale_embedding_dim=8,  # Compact scale embeddings
        
        # Fusion configuration
        attention_dim=32,
        fusion_channels=128,
    ),
    
    # Standard middle encoder (compatible with output)
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=73,  # 64 features + 8 scale_emb + 1 scale_id
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
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

# Training configuration (identical to other configs for fair comparison)
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=5)
train_dataloader = dict(batch_size=1)  # Start with batch size 1, may increase due to memory efficiency

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
work_dir = './work_dirs/importance_guided_multi_scale'

# Architecture Summary:
# =====================
# 
# Memory Efficiency Features:
# ---------------------------
# 1. Point Importance Network: 3-layer MLP (4→64→32→16→1) filters out 30% least important points
# 2. Lightweight VFE: Reduced channels (32, 64) vs standard (64, 128)
# 3. Compact Scale Embeddings: 8-dim vs 16-dim in full version
# 4. Shared Processing: Same VFE for all scales with scale ID embedding
# 5. Efficient Attention: 32-dim attention vs full feature dimension
#
# Expected Benefits:
# ------------------
# - 30% reduction in point processing (70% points kept)
# - Lighter VFE networks (50% channel reduction)  
# - Shared computations across scales
# - Lower memory footprint for multi-scale processing
# - Maintained accuracy through importance-based selection
#
# Comparison with Previous Configs:
# ---------------------------------
# - Vanilla SECOND: 128→128 channels, all points
# - Advanced Multi-Scale: 81→256 channels, all points, heavy attention
# - Importance-Guided: 73→128 channels, 70% points, lightweight processing
#
# To run training:
# python tools/train.py configs/importance_guided_multi_scale_second.py
#
# Expected memory usage: Significantly lower than advanced version, similar to vanilla
