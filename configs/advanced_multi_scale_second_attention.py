"""
Advanced Multi-Scale SECOND Configuration with Attention
======================================================

Configuration for the advanced multi-scale VFE module with attention mechanisms.
Implements state-of-the-art multi-scale processing for 3D object detection.

Features:
- Three voxel resolutions (0.05m, 0.1m, 0.2m)
- Separate VFE for each scale
- Attention-based importance weighting
- Learnable scale embeddings
- Enhanced middle encoder

Author: PhD Research Implementation
Date: August 3, 2025
"""

# Import required modules to ensure registration
import mmdet3d  # This ensures all modules are properly registered
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# Import models to ensure they are registered
from mmdet3d.models.detectors.voxelnet import VoxelNet
from mmdet3d.models.backbones.second import SECOND
from mmdet3d.models.necks.second_fpn import SECONDFPN
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead

# Import all MMDetection3D models to ensure registration
import mmdet3d.models

# Standalone configuration without _base_ inheritance

# Dataset settings
dataset_type = 'KittiDataset'
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
class_names = ['Car']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None

# Point cloud range
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration
model = dict(
    type='VoxelNet',
    
    # Advanced Multi-Scale VFE with Attention
    voxel_encoder=dict(
        type='MultiScaleVFEWithAttention',
        point_cloud_range=point_cloud_range,
        max_num_points=5,  # Points per voxel
        max_voxels=(12000, 30000),  # Max voxels (train, test)
        voxel_scales=[0.05, 0.1, 0.2],  # Fine, medium, coarse
        feature_dim=64,  # Features per scale
        scale_embedding_dim=16,  # Scale embedding dimension
        attention_dim=32,  # Attention mechanism dimension
    ),
    
    # Enhanced Middle Encoder for attention features
    middle_encoder=dict(
        type='EnhancedMultiScaleParallelMiddleEncoder',
        in_channels=81,  # 64 (features) + 16 (scale_emb) + 1 (scale_id)
        output_channels=256,
        sparse_shape=[41, 1600, 1408],
    ),
    
    # Standard SECOND backbone
    backbone=dict(
        type='SECOND',
        in_channels=256,  # From enhanced middle encoder
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    # Standard neck
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    
    # Standard bbox head
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -40, -0.6, 70.4, 40, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2
        )
    ),
    
    # Training configuration
    train_cfg=dict(
        assigner=[
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1
            )
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
    
    # Testing configuration
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

# Data configuration for multi-scale processing
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['Car']),
    dict(type='PointShuffle'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]
            ),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Dataset configuration
train_dataloader = dict(
    batch_size=4,  # Reduced for memory efficiency
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

# Optimizer configuration - using conservative settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),  # Conservative LR
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=40
    )
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=5
)

# Evaluation configuration
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation metrics
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args
)
test_evaluator = val_evaluator

# Default hooks for enhanced training
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,
        max_keep_ckpts=5,
        save_best='KITTI/Car_3D_moderate_strict',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Visualization configuration
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type=Det3DLocalVisualizer, 
    vis_backends=vis_backends, 
    name='visualizer'
)

# Logging configuration
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# Runtime configuration
randomness = dict(seed=0, deterministic=False)

# Work directory
work_dir = './work_dirs/advanced_multi_scale_second_attention'

print("🚀 Advanced Multi-Scale SECOND with Attention Configuration Loaded")
print("📊 Features: 3 scales, attention mechanism, learnable embeddings")
print("🎯 Expected performance: Enhanced accuracy with intelligent scale weighting")
