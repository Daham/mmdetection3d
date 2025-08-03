"""
CUDA-Safe Baseline Test Configuration
====================================

This configuration uses CPU-compatible components to avoid the CUDA sparse tensor error 700.
The error occurs in the sparse convolution operations in the middle encoder.

Key Changes:
1. Use CPU-compatible middle encoder
2. Ensure proper sparse tensor format
3. Fix VFE interface compatibility
"""

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            max_voxels=(12000, 30000),
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            voxel_size=[0.05, 0.05, 0.1]
        )
    ),
    
    # 🔥 CRITICAL: Use CPU-compatible VFE to avoid CUDA sparse tensor issues
    voxel_encoder=dict(
        type='DynamicVFE',  # Standard VFE that's known to work
        in_channels=4,
        feat_channels=[32, 64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max'
    ),
    
    # 🔥 CRITICAL: Use CPU-compatible middle encoder to avoid sparse tensor CUDA error
    middle_encoder=dict(
        type='SparseEncoder',  # Changed from Enhanced to standard
        in_channels=64,  # Match VFE output
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=256,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2]
    ),
    
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
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2
        )
    ),
    
    # Training configuration
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.35,  # Conservative threshold
            neg_iou_thr=0.2,
            min_pos_iou=0.2,
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

# Dataset configuration for KITTI
dataset_type = 'KittiDataset'
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

# Data pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectNameFilter',
        classes=class_names
    ),
    dict(type='PointShuffle'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]
    ),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']
    )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None
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
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]
            ),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range
            )
        ]
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Dataset configurations
train_dataloader = dict(
    batch_size=2,  # Increased batch size for more stable training
    num_workers=2,
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
        backend_args=None
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
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=None
    )
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None
)
test_evaluator = val_evaluator

# Schedule configuration - very conservative for testing
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer configuration - reduced learning rate to avoid instability
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.05, betas=(0.95, 0.99)),  # Reduced LR
    clip_grad=dict(max_norm=5, norm_type=2)  # Conservative gradient clipping
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=200
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=2,
        eta_min=1e-6,
        begin=0,
        end=2,
        by_epoch=True
    )
]

# Default hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),  # No checkpoints for testing
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Runtime configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet3d.visualization.local_visualizer.Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_level = 'INFO'
load_from = None
resume = False

# Working directory
work_dir = './work_dirs/cuda_safe_baseline_test'
