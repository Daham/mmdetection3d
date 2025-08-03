"""
CUDA Error 700 Debug Configuration
==================================

This configuration helps debug and fix the CUDA error 700 by:
1. Using debug middle encoder with detailed logging
2. Validating sparse tensor coordinates
3. Providing fallback mechanisms

The CUDA error occurs at: sparse_indice.cu:120 cuda execution failed with error 700
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
    
    # Use standard VFE that's known to work
    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[32, 64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max'
    ),
    
    # 🔥 CRITICAL: Use debug encoder to identify CUDA issue
    middle_encoder=dict(
        type='DebugSparseEncoder',
        in_channels=64,
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
    
    train_cfg=dict(
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

# Dataset configuration
dataset_type = 'KittiDataset'
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

# Simple training pipeline to minimize variables
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
    # Minimal augmentations to reduce complexity
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],  # No rotation
        scale_ratio_range=[1.0, 1.0],  # No scaling
        translation_std=[0, 0, 0]  # No translation
    ),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.0  # No flipping
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

# Dataset configurations with minimal batch size
train_dataloader = dict(
    batch_size=1,  # Single sample to isolate issue
    num_workers=1,  # Single worker to avoid race conditions
    persistent_workers=False,  # Disable persistence to avoid memory issues
    sampler=dict(type='DefaultSampler', shuffle=False),  # No shuffling for consistent debugging
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
    persistent_workers=False,
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

# Minimal training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Conservative optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01, betas=(0.9, 0.99)),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=10
    )
]

# Hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),  # Log every iteration
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Runtime configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

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
work_dir = './work_dirs/cuda_error_debug'
