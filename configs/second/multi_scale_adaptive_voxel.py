# 🎯 TRUE MULTI-SCALE ADAPTIVE VOXELIZATION - Your Revolutionary Architecture
_base_ = [
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🔬 COMPLETE MODEL WITH YOUR BRILLIANT MULTI-SCALE ARCHITECTURE
model = dict(
    type='VoxelNet',
    
    # 🔬 RAW POINTS INPUT: No fixed voxelization at preprocessor level
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False,  # Disable standard voxelization - we handle this adaptively
        voxel_type='hard',
        voxel_layer=None,  # No fixed voxelization layer
        mean=[0, 0, 0],
        std=[1, 1, 1],
    ),
    
    # 🌟 MINIMAL PASSTHROUGH VOXEL ENCODER
    voxel_encoder=dict(
        type='PassthroughVoxelEncoder'
    ),
    
    # 🌉 MULTI-SCALE MIDDLE ENCODER BRIDGE: Complete pipeline
    middle_encoder=dict(
        type='AdaptiveMultiScaleBridge',
        
        # 🔧 VOXEL ENCODER CONFIG: Multi-scale adaptive voxelization
        voxel_encoder_config=dict(
            type='MultiScaleAdaptiveVoxelEncoder',
            point_cloud_range=point_cloud_range,
            
            # 🎯 BASE CONFIGURATION
            base_voxel_size=[0.05, 0.05, 0.1],      # Standard voxel size
            max_num_points=5,                        # Points per voxel
            max_voxels=(12000, 30000),              # Training and test limits
            
            # 🎯 SCALE FACTORS FOR MULTI-SCALE PROCESSING
            fine_scale=0.5,      # 2x finer: [0.025, 0.025, 0.05]
            medium_scale=1.0,    # Base scale: [0.05, 0.05, 0.1]  
            coarse_scale=2.0,    # 2x coarser: [0.1, 0.1, 0.2]
            
            # 🧠 IMPORTANCE PREDICTOR: Neural network for scale assignment
            importance_predictor=dict(
                input_channels=4,                    # x, y, z, intensity
                hidden_channels=[16, 8],             # Lightweight network
                num_classes=3,                       # fine, medium, coarse
                dropout_rate=0.1
            ),
            
            # 🎯 FEATURE PROCESSING: Parallel extraction and fusion
            feature_fusion=dict(
                type='MultiScaleFeatureFusion',
                input_channels=[5, 5, 5],            # Features per scale
                output_channels=64,                   # Unified output
                fusion_method='attention'             # Attention-based fusion
            )
        ),
        
        # 🌐 MIDDLE ENCODER CONFIG: Sparse convolution processing
        middle_encoder_config=dict(
            type='SparseEncoder',
            in_channels=4,
            sparse_shape=[41, 1600, 1408],
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
            block_type='basicblock',
        )
    ),
    
    # 🎯 BACKBONE: Standard SECOND architecture  
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    # 🎯 NECK: Standard SECOND FPN
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]
    ),
    
    # 🎯 DETECTION HEAD: Standard Anchor3DHead for Car detection
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,  # Only Car class
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],  # Only Car
            sizes=[[3.9, 1.6, 1.56]],  # Car size
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
            beta=1.0 / 9.0, 
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
        assigner=dict(  # for Car
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1,
        ),
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

# Training configuration
train_dataloader = dict(
    batch_size=2,  # Adjusted for memory efficiency with multi-scale processing
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,  # No dataset repetition
        dataset=dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
                dict(type='ObjectSample', db_sampler=dict(
                    data_root='data/kitti/',
                    info_path='data/kitti/kitti_dbinfos_train.pkl',
                    rate=1.0,
                    prepare=dict(
                        filter_by_difficulty=[-1],
                        filter_by_min_points=dict(Car=5)
                    ),
                    classes=['Car'],
                    sample_groups=dict(Car=15)
                )),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(type='GlobalRotScaleTrans', 
                     rot_range=[-0.78539816, 0.78539816],
                     scale_ratio_range=[0.95, 1.05],
                     translation_std=[0, 0, 0]),
                dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='ObjectNameFilter', classes=class_names),
                dict(type='PointShuffle'),
                dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            metainfo=dict(classes=class_names),
            box_type_3d='LiDAR'
        )
    )
)

# Validation configuration
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR'
    )
)

# Test configuration
test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox'
)
test_evaluator = val_evaluator

# Optimizer configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'voxel_encoder': dict(lr_mult=0.1),  # Lower learning rate for voxel encoder
            'importance_predictor': dict(lr_mult=2.0),  # Higher learning rate for importance predictor
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)

# Learning rate configuration
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=80,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

# Runtime configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
