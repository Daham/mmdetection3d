"""
🔬 CLEAN ADAPTIVE VOXELIZATION RESEARCH CONFIGURATION
No base configs - completely standalone to avoid conflicts
"""

# Dataset settings
dataset_type = 'KittiDataset'
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
class_names = ['Car']

# Model configuration
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=35,
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            voxel_size=[0.16, 0.16, 4.0],
            max_voxels=(16000, 40000))),
    
    # 🔬 CORE RESEARCH: Adaptive learnable voxel encoder
    voxel_encoder=dict(
        type='AdaptiveLearnableVoxelLayer',
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        base_voxel_size=[0.16, 0.16, 4.0],
        max_num_points=35,
        max_voxels=(16000, 40000),
        voxel_size_scale_range=(0.5, 2.0),
        importance_threshold=0.5),
    
    # 🔬 ADAPTIVE: Custom middle encoder with sparse convolutions
    middle_encoder=dict(
        type='AdaptiveVoxelEncoder',
        in_channels=4,
        out_channels=128,
        sparse_shape=[41, 1600, 1408]),
    
    backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[64, 128],
        layer_nums=[3, 5],
        layer_strides=[2, 2]),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],
        out_channels=[128, 128],
        upsample_strides=[1, 2]),
    
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
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
    # Training configuration  
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# Training configuration - CLEAN ITERATION-ONLY SETUP
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=5,
    val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Dataset pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
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
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -40, -3, 70.4, 40, 1])
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])]

# Dataloader settings
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            metainfo=dict(classes=class_names),
            box_type_3d='LiDAR')))

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
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# Optimization
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2),
    dict(
        type='CosineAnnealingLR',
        T_max=3,
        by_epoch=False,
        begin=2,
        end=5)
]

# Hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Scope and imports
default_scope = 'mmdet3d'

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Visualization - use basic visualizer to avoid registration issues
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet3d.Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Runtime
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = 'work_dirs/adaptive_research'

print("🔬 CLEAN ADAPTIVE VOXELIZATION RESEARCH CONFIG - NO CONFLICTS!")
print("✅ Adaptive voxel sizes: LEARNABLE PARAMETERS")
print("✅ Importance prediction: IMPLEMENTED") 
print("✅ Memory efficiency: IMPLEMENTED")
print("⚡ Testing: 5 iterations to verify research implementation")
