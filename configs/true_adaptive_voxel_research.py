# 🔬 TRUE ADAPTIVE VOXELIZATION CONFIG - Real Voxel Size Learning
# This config bypasses the standard data preprocessor to enable true adaptive voxel sizes

_base_ = [
    './_base_/datasets/kitti-3d-3class.py', 
    './_base_/schedules/cyclic-40e.py', 
    './_base_/default_runtime.py',
]

# 📊 Data pipeline without pre-voxelization  
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)

# 🔬 BREAKTHROUGH: NO PRE-VOXELIZATION IN DATA PREPROCESSOR
model = dict(
    type='VoxelNet',
    
    # 🔧 CRITICAL: Disable pre-voxelization to enable true adaptive voxelization
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False,  # 🚨 DISABLE standard voxelization!
        # No voxel_layer = no pre-voxelization = true adaptive voxel research!
    ),
    
    # 🔬 PURE ADAPTIVE VOXEL LAYER - True variable voxel sizes!
    voxel_encoder=dict(
        type='PureAdaptiveVoxelLayer',
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        base_voxel_size=[0.16, 0.16, 4.0],  # Learnable parameters!
        max_num_points=20,
        max_voxels=(8000, 20000),
        voxel_size_scale_range=(0.3, 3.0),  # Wide range for research
        importance_threshold=0.4),
    
    # 🔧 STANDARD SPARSE ENCODER - Receives true adaptive voxels
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,  # Raw point features from adaptive voxelization
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='conv_module'),
    
    backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2]),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4]),
    
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            anchor_3d_sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            anchor_bottom_heights=[-0.6, -0.6, -1.78],
            align_center=False,
            feature_map_sizes=None,
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            rotations=[0, 1.57]),
        
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
    train_cfg=dict(
        assigner=[
            dict(type='MaxIoUAssigner',
                 iou_calculator=dict(type='BboxOverlapsNearest3D'),
                 pos_iou_thr=0.6, neg_iou_thr=0.45, min_pos_iou=0.45,
                 ignore_iof_thr=-1),
            dict(type='MaxIoUAssigner',
                 iou_calculator=dict(type='BboxOverlapsNearest3D'),
                 pos_iou_thr=0.35, neg_iou_thr=0.2, min_pos_iou=0.2,
                 ignore_iof_thr=-1),
            dict(type='MaxIoUAssigner',
                 iou_calculator=dict(type='BboxOverlapsNearest3D'),
                 pos_iou_thr=0.6, neg_iou_thr=0.45, min_pos_iou=0.45,
                 ignore_iof_thr=-1)],
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

# 🔬 RESEARCH DATASET: Raw points without pre-voxelization
train_dataloader = dict(
    batch_size=1,  # Conservative for memory with true adaptive voxelization
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_train.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            # 🔬 NO VOXELIZATION HERE - Raw points go to model!
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(type='GlobalRotScaleTrans',
                 rot_range=[-0.15707963267, 0.15707963267],
                 scale_ratio_range=[0.95, 1.05]),
            dict(type='PointsRangeFilter', point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
            dict(type='ObjectRangeFilter', point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
            dict(type='PointShuffle'),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
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
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

# 🔬 RESEARCH-FRIENDLY OPTIMIZER
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    # 🔬 IMPORTANT: Include voxel size parameters in optimization!
    paramwise_cfg=dict(custom_keys={
        'voxel_encoder.base_voxel_size': dict(lr_mult=0.1),  # Lower LR for voxel sizes
        'voxel_encoder.fine_scale': dict(lr_mult=0.1),
        'voxel_encoder.coarse_scale': dict(lr_mult=0.1)
    }),
    clip_grad=dict(max_norm=10, norm_type=2))

# 🔬 CONSERVATIVE SCHEDULER
param_scheduler = [
    dict(type='CyclicLR',
         target_ratio=(5, 1e-4),  # Gentler for research
         cyclic_times=1,
         step_ratio_up=0.4)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=5)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 🔬 VALIDATION & TESTING
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')

test_evaluator = val_evaluator

# 🔬 RESEARCH LOGGING - Track voxel size learning!
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggingHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# 🔬 RESEARCH LOGGING FOR VOXEL SIZE TRACKING
custom_hooks = [
    dict(type='AdaptiveVoxelSizeLogger',
         log_interval=100,
         track_voxel_parameters=True,
         save_voxelization_stats=True)
]
