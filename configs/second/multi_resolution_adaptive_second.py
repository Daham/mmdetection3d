# Multi-Resolution Adaptive SECOND Configuration
# This config uses true variable voxel size processing with MultiResolutionSparseEncoder

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-20e.py',  # Longer training for complex model
    '../_base_/default_runtime.py'
]

# Custom imports for multi-resolution modules
custom_imports = dict(
    imports=[
        'mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe',
        'mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder'
    ],
    allow_failed_imports=False)

# Dataset configuration
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Multi-resolution adaptive model configuration
model = dict(
    # Enhanced Adaptive VFE with multi-resolution support
    voxel_encoder=dict(
        type='EnhancedAdaptiveVFE',
        in_channels=4,
        feat_channels=[64],  # Increased channels for richer features
        with_distance=True,  # Enable distance features for better adaptation
        voxel_size=(0.05, 0.05, 0.1),  # Base voxel size
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='density',  # Use density-based adaptation
        num_scales=3,
        provide_multi_res_info=True  # Enable multi-resolution information
    ),
    
    # Multi-Resolution Sparse Encoder - THE KEY INNOVATION!
    middle_encoder=dict(
        type='MultiResolutionSparseEncoder',
        base_voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        resolution_levels=[0.5, 1.0, 2.0],  # Fine, medium, coarse scales
        in_channels=64,  # Must match EnhancedAdaptiveVFE output
        out_channels=256,  # Standard SECOND middle encoder output
        assignment_threshold=0.1,  # Threshold for resolution assignment
        fusion_method='weighted_concat'  # How to fuse multi-resolution features
    ),
    
    # Standard SECOND backbone (unchanged)
    backbone=dict(
        type='SECOND',
        in_channels=256,  # Must match MultiResolutionSparseEncoder output
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    
    # Configure for single class (Car only) for focused evaluation
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],  # Car size
            rotations=[0, 1.57],
            reshape_out=True
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        )
    ),
    
    # Training configuration optimized for multi-resolution learning
    train_cfg=dict(
        _delete_=True,
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

# Training schedule - longer for complex multi-resolution model
train_cfg = dict(max_epochs=20, val_interval=2)

# Optimizer configuration - lower learning rate for stable multi-resolution training
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0001,  # Lower learning rate for stable training
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=10, norm_type=2)  # Gradient clipping for stability
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=20
    )
]

# Data loading configuration - smaller batch size for memory efficiency
train_dataloader = dict(
    batch_size=2,  # Reduced batch size due to multi-resolution complexity
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
                dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(type='ObjectNameFilter', classes=['Car']),
                dict(type='PointShuffle'),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5
                ),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]
                ),
                dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']
                )
            ],
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            metainfo=dict(classes=['Car']),
            box_type_3d='LiDAR'
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
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
            dict(type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
            dict(type='ObjectNameFilter', classes=['Car']),
            dict(
                type='Pack3DDetInputs',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']
            )
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['Car']),
        box_type_3d='LiDAR'
    )
)

# Evaluation configuration
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    format_only=False
)

# Logging configuration for monitoring multi-resolution training
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=2,  # Save every 2 epochs
        max_keep_ckpts=5,
        save_best='auto'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Experiment tracking
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

# Work directory for this experiment
work_dir = './work_dirs/multi_resolution_adaptive_second'
