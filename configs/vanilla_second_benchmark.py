"""
Vanilla SECOND Configuration for Benchmarking
============================================

This configuration provides a baseline vanilla SECOND implementation
with identical settings to the advanced multi-scale version for 
direct performance and memory efficiency comparison.

Comparison Features:
- Same batch size, epochs, and optimizer settings
- Same data preprocessing and voxel settings  
- Same learning rate schedule
- Standard VFE vs Multi-Scale VFE with Attention
- Standard middle encoder vs Enhanced middle encoder

Author: PhD Research Benchmarking
Date: August 3, 2025
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-car.py',
    './_base_/schedules/cyclic-2e.py',
    './_base_/default_runtime.py'
]

# Data root configuration (same as advanced version)
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Point cloud range and voxel settings (identical to advanced version)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration with standard SECOND components
model = dict(
    # Keep VoxelNet type from base config
    type='VoxelNet',
    
    # Standard data preprocessor (same voxel settings as advanced version)
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],  # Same base voxel size
            max_voxels=(12000, 30000)
        )
    ),
    
    # Standard VFE (vanilla SECOND)
    voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=[0.05, 0.05, 0.1],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max'
    ),
    
    # Standard middle encoder (vanilla SECOND)
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=128,  # Standard HardVFE output
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    # Same bbox head configuration as advanced version
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            ranges=[[0, -40, -0.6, 70.4, 40, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False
        )
    ),
    
    # Same train configuration as advanced version
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

# Training configuration (identical to advanced version)
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=5)
train_dataloader = dict(batch_size=1)  # Same batch size for fair comparison

# Optimizer configuration (identical to advanced version)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate scheduler (identical to advanced version)
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

# Work directory for benchmark results
work_dir = './work_dirs/vanilla_second_benchmark'

# Benchmarking notes:
# ==================
# Memory Usage Comparison:
# - Vanilla SECOND: HardVFE (128 channels) + SparseEncoder (128 channels)
# - Advanced Multi-Scale: MultiScaleVFEWithAttention (81 channels) + EnhancedMiddleEncoder (256 channels)
# 
# Expected Differences:
# - VFE: Vanilla uses single-scale HardVFE vs Multi-scale with attention
# - Middle Encoder: Standard sparse encoder vs enhanced parallel processing
# - Feature Channels: 128 → 128 (vanilla) vs 81 → 256 (advanced)
# - Memory: Vanilla should use less GPU memory but potentially less feature richness
# - Performance: Advanced should have better accuracy but higher computational cost
#
# To run benchmark:
# python tools/train.py configs/vanilla_second_benchmark.py
#
# To compare memory usage:
# Monitor GPU memory during training of both configurations
