# Complete Multi-Resolution Adaptive Voxelization Training Config
# This config demonstrates the full adaptive voxelization approach with multi-resolution sparse convolution

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Custom imports for multi-resolution adaptive modules
custom_imports = dict(
    imports=[
        'mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe',
        'mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder'
    ],
    allow_failed_imports=False)

# Dataset configuration
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Complete model configuration with multi-resolution adaptive voxelization
model = dict(
    # Enhanced Adaptive VFE that provides multi-resolution information
    voxel_encoder=dict(
        type='EnhancedAdaptiveVFE',
        in_channels=4,
        feat_channels=[64, 128],  # Richer features for multi-resolution
        with_distance=True,  # Include distance information
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='multi_scale',  # Use multi-scale adaptation
        num_scales=3,
        provide_multi_res_info=True),
    
    # Multi-Resolution Sparse Encoder for processing variable voxel sizes
    middle_encoder=dict(
        type='MultiResolutionSparseEncoder',
        base_voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        resolution_levels=[0.5, 1.0, 2.0],  # Fine, base, coarse resolutions
        in_channels=128,  # Match VFE output
        out_channels=256,  # Rich multi-resolution features
        assignment_threshold=0.1,
        fusion_method='attention'),  # Use attention-based fusion
    
    # Backbone to process fused multi-resolution features
    backbone=dict(
        type='SecondFPN',
        in_channels=[256],  # Match multi-resolution encoder output
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    # Neck for feature aggregation
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    # Detection head for single class (Car)
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,  # Sum of neck outputs
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],  # Car size
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
    # Training configuration
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    
    # Testing configuration
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# Training data configuration for adaptive multi-resolution learning
train_dataloader = dict(
    batch_size=2,  # Reduced batch size due to multi-resolution processing
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type='GlobalRotScaleTrans', 
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
            dict(type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
            dict(type='PointShuffle'),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=False,
        metainfo=dict(classes=['Car']),
        box_type_3d='LiDAR'))

# Validation data
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['Car']),
        box_type_3d='LiDAR'))

# Test data
test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')

test_evaluator = val_evaluator

# Optimizer configuration - reduced learning rate for multi-resolution stability
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=10, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.33333333, 
        by_epoch=False, 
        begin=0, 
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-7,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime configuration for multi-resolution monitoring
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Visualization configuration
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Logging configuration
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# Auto scaling configuration
auto_scale_lr = dict(enable=False, base_batch_size=16)
