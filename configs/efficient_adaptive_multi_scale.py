"""
🚀 EFFICIENT Multi-Scale Adaptive Voxelization Configuration
============================================================

CRITICAL PhD REQUIREMENT: "Separate tensors for different voxel sizes" processed in "parallel sparse convolution networks"

Performance Targets:
- Match vanilla SECOND baseline: 0.8-1.8s/iter, ~797MB memory
- Maintain all PhD research requirements
- Use high-performance vectorized operations

This config uses EfficientMultiScaleParallelMiddleEncoder for maximum performance
while preserving all revolutionary multi-scale adaptive voxelization features.

Author: PhD Research Implementation  
Date: August 3, 2025
Status: HIGH-PERFORMANCE SOLUTION
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-car.py',
    './_base_/schedules/cyclic-40e.py',
    './_base_/default_runtime.py'
]

# Custom classes
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.optimized_multi_scale_adaptive_voxel',
             'mmdet3d.models.middle_encoders.efficient_multi_scale_parallel_middle_encoder'],
    allow_failed_imports=False)

# High-performance model configuration
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False,
        voxel_layer=None
    ),
    voxel_encoder=dict(
        type='OptimizedMultiScaleAdaptiveVoxelEncoder',
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        base_voxel_size=[0.05, 0.05, 0.1],
        fine_scale=0.5,      # 0.025m voxels for fine details
        medium_scale=1.0,    # 0.05m base voxels  
        coarse_scale=2.0,    # 0.1m voxels for coarse features
        max_num_points=5,
        max_voxels=(12000, 30000),  # Optimized voxel limits
        importance_channels=64
    ),
    middle_encoder=dict(
        type='EfficientMultiScaleParallelMiddleEncoder',  # 🚀 HIGH-PERFORMANCE ENCODER
        in_channels=64,
        output_channels=128,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')
    ),
    backbone=dict(
        type='SECOND',
        in_channels=128,  # Matches middle encoder output
        layer_nums=[3, 5],
        layer_strides=[2, 2],
        out_channels=[64, 128]
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],
        out_channels=[128, 128],
        upsample_strides=[1, 2]
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50
    )
)

# Optimized training configuration for maximum performance
train_dataloader = dict(
    batch_size=1,  # Efficient memory usage
    num_workers=2,
    persistent_workers=True,
    pin_memory=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True
)

# Fast training schedule for quick validation
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=3,  # Quick test
    val_interval=1
)

# Optimized optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.002, betas=(0.9, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        by_epoch=False,
        begin=0,
        end=100
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min=0.0005,
        begin=0,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Default hooks for efficient logging
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),  # Log every 20 iterations
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Performance monitoring
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Output directory
work_dir = './work_dirs/efficient_adaptive_multi_scale'
