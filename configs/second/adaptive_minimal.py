"""
Minimal Adaptive Voxelization Config - No Database Sampling

This config removes database sampling dependencies so you can start training
immediately even without the kitti_dbinfos_train.pkl file.
Use this for initial testing, then switch to full config once dataset is prepared.
"""

# =============================================================================
# DATASET CONFIGURATION - UPDATE THIS FOR YOUR MACHINE
# =============================================================================
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
dataset_type = 'KittiDataset'
class_names = ['Car']
metainfo = dict(classes=['Car'])
input_modality = dict(use_lidar=True, use_camera=False)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# =============================================================================
# ADAPTIVE VOXELIZATION MODEL (Same as main config)
# =============================================================================
model = dict(
    type='AdaptiveVoxelNet',
    
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1])),
    
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        num_features=4,
        min_voxel_size=0.05,
        max_voxel_size=0.50,
        initial_bias=0.2,
        voxel_predictor_hidden=128,
        spatial_encoding_dim=64,
        voxel_aware_hidden=128,
    ),
    
    middle_encoder=dict(
        type='AdaptiveSparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=['conv', 'norm', 'act'],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=[16, 32, 64, 64, 64, 64],
        encoder_paddings=[1, 1, 1, 1, 1, 1],
        block_type='conv_module',
        num_size_groups=4,
        size_group_ranges=[
            (0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.50),
        ],
        fusion_type='attention',
    ),
    
    backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2]),
    
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6]],
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
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2)),
    
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
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

# =============================================================================
# SIMPLIFIED TRAINING PIPELINE (No Database Sampling)
# =============================================================================
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # Removed ObjectSample (database sampling) to avoid dependency on .pkl files
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

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
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# =============================================================================
# DATASETS (Simplified - using train/val split from infos files only)
# =============================================================================
train_dataloader = dict(
    batch_size=2,  # Smaller batch for initial testing
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',  # You'll need this file
        data_prefix=dict(pts='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',  # You'll need this file
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=5)  # Less frequent validation
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'voxel_encoder': dict(lr_mult=1.5),
            'middle_encoder.size_pathways': dict(lr_mult=1.2),
            'middle_encoder.fusion': dict(lr_mult=1.3),
        }),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=0.018,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=1.8e-7,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================
default_scope = 'mmdet3d'
work_dir = './work_dirs/adaptive_minimal'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

print("🚀 Minimal Adaptive Voxelization Config Loaded (No Database Sampling)")
print("📝 Note: This config skips database sampling for easier setup")
print("🔧 To enable full data augmentation, prepare kitti_dbinfos_train.pkl")
