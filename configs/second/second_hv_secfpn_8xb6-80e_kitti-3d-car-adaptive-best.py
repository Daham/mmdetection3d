# configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car-adaptive-best.py
# OPTIMAL CONFIG FOR ADAPTIVE VOXELIZATION
# Using AdaptiveSparseEncoderV3Simple as the best balance of performance and adaptivity

# Import custom modules to ensure registration
from mmdet3d.models.voxel_encoders.adaptive_vfe import AdaptiveVFE
from mmdet3d.models.middle_encoders.adaptive_sparse_encoder_v3 import AdaptiveSparseEncoderV3Simple

_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)

# Model settings - OPTIMAL ADAPTIVE CONFIGURATION
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000))),
    
    # BEST: AdaptiveVFE with density-based adaptation
    voxel_encoder=dict(
        type='AdaptiveVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.05, 0.05, 0.1],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        # Adaptive settings
        adaptive_type='density_based',  # Best for real-world performance
        base_voxel_size=[0.05, 0.05, 0.1],
        size_bounds=[0.5, 2.0],  # Conservative bounds for stability
        learning_rate=0.001,  # Lower LR for stable adaptation
        density_threshold=0.5),
    
    # BEST: AdaptiveSparseEncoderV3Simple - optimal balance
    middle_encoder=dict(
        type='AdaptiveSparseEncoderV3Simple',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        # Adaptive settings
        adaptive_channel_boost=64),  # Moderate boost for stability
    
    # Standard SECOND backbone - proven performance
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    # Standard neck
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    # Standard detection head
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
    # Training settings
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1),
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

# Training settings
train_dataloader = dict(batch_size=6, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader

# Optimizer with adaptive-friendly settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.01),  # Lower LR for adaptive
    paramwise_cfg=dict(custom_keys={
        'voxel_encoder.size_factors': dict(lr_mult=0.1),  # Even lower for adaptive params
        'middle_encoder.size_processor': dict(lr_mult=0.1)
    }),
    clip_grad=dict(max_norm=10, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(type='CyclicLR', 
         target_ratio=(10, 1e-4), 
         cyclic_times=1, 
         step_ratio_up=0.4,
         by_epoch=False,
         begin=0,
         end=7330),
    dict(type='CyclicLR',
         target_ratio=(1e-4, 1e-7),
         cyclic_times=1,
         step_ratio_up=0.0,
         by_epoch=False,
         begin=7330,
         end=12544)
]

# Evaluation
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Custom hooks for adaptive monitoring - commented out until implemented
# custom_hooks = [
#     dict(type='AdaptiveMonitorHook',
#          log_interval=100,
#          monitor_size_factors=True,
#          monitor_middle_encoder=True)
# ]

load_from = None
resume_from = None

# Explicit training loop configuration for 2 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
