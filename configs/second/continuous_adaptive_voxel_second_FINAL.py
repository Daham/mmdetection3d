# 🌊 Continuous Adaptive Voxel SECOND Configuration
# Advanced continuous voxel size prediction with soft interpolation

_base_ = [
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/models/second.py',
    '../_base_/schedules/cyclic-40e.py',
    '../_base_/default_runtime.py',
]

# Point cloud range
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=['Car'])

# 🌊 CONTINUOUS ADAPTIVE VOXEL CONFIGURATION
voxel_layer = dict(
    _delete_=True,  # Remove original voxel_layer
    type='MultiScaleDynamicVoxelize',
    voxel_scales=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0],  # 10 scales
    max_num_points=5,
    point_cloud_range=point_cloud_range
)

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=voxel_layer),
    
    # 🌊 CONTINUOUS ADAPTIVE VFE 
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=(0.05, 0.05, 0.1),  # Default reference size
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',
        legacy=False,
        
        # 🌊 NEW: Continuous prediction parameters
        scale_net_cfg=dict(
            type='ScaleNet',
            in_channels=4,  # x, y, z, intensity
            hidden_dims=[128, 64, 32],  # Larger network for continuous prediction
            num_scales=10,  # Number of discrete reference scales
            temperature=3.0,  # Lower initial temperature for continuous mode
            dropout_rate=0.03,  # Reduced dropout for better regression
            
            # 🌊 CONTINUOUS MODE SETTINGS
            continuous_mode=True,  # Enable continuous prediction!
            min_voxel_size=0.01,  # Minimum voxel size (meters)
            max_voxel_size=1.0,   # Maximum voxel size (meters)
            interpolation_neighbors=4  # Use 4 nearest scales for interpolation
        ),
        
        # Enhanced multi-scale processing
        multi_scale_cfg=dict(
            num_scales=10,
            scale_factor=1.5,  # Smaller factor for smoother transitions
            attention_cfg=dict(
                embed_dims=64,
                num_heads=8,
                dropout=0.05,
                cross_scale_attention=True  # Enable cross-scale feature interaction
            )
        )
    ),
    
    # Enhanced middle encoder for better feature processing
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
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
            reshape_out=False),
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
    
    # Model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # For cars
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
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

# Enhanced training configuration for continuous prediction
train_dataloader = dict(
    batch_size=2,  # Smaller batch for stable continuous training
    num_workers=4,
    dataset=dict(
        type='RepeatDataset',
        times=2,
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
                        filter_by_min_points=dict(Car=5)),
                    classes=['Car'],
                    sample_groups=dict(Car=15))),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(type='GlobalRotScaleTrans',
                     rot_range=[-0.15707963267, 0.15707963267],
                     scale_ratio_range=[0.95, 1.05],
                     translation_std=[0, 0, 0]),
                dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='ObjectNameFilter', classes=['Car']),
                dict(type='PointShuffle'),
                dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=None)))

# Enhanced validation configuration
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=None))

test_dataloader = val_dataloader

# 🌊 CONTINUOUS LEARNING OPTIMIZATIONS
# Custom optimizer settings for continuous prediction
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0003, betas=(0.95, 0.99), weight_decay=0.05),  # Lower LR for stable continuous training
    paramwise_cfg=dict(
        custom_keys={
            'continuous_head': dict(lr_mult=1.5),  # Higher LR for continuous head
            'confidence_head': dict(lr_mult=1.2),  # Slightly higher for confidence
            'scale_predictor': dict(lr_mult=0.8),   # Lower LR for discrete predictor
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2))

# Enhanced learning rate schedule for continuous prediction
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=40,
        T_max=40,
        eta_min_ratio=1e-4,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Extended training epochs for continuous learning
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 🌊 CONTINUOUS PREDICTION LOGGING
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5),
    logger=dict(type='LoggingHook', interval=50),
    visualization=dict(type='Det3DVisualizationHook'))

# Custom evaluation for continuous prediction analysis
val_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None)

test_evaluator = val_evaluator

# Enable automatic mixed precision for efficiency
fp16 = dict(loss_scale=32.)

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Load pretrained backbone if available
load_from = None
resume = False

# Continuous prediction experiment settings
experiment_name = 'continuous_adaptive_voxel_second'
work_dir = f'./work_dirs/{experiment_name}'

# 🌊 Configuration Summary
print("🌊 CONTINUOUS ADAPTIVE VOXEL SECOND CONFIGURATION")
print("=" * 60)
print("🎯 Continuous Prediction: ENABLED")
print(f"📏 Voxel Size Range: 0.01m - 1.0m")
print(f"🔢 Reference Scales: 10 scales")
print(f"🤝 Interpolation Neighbors: 4")
print(f"🧠 Network: Deeper (128→64→32)")
print(f"⚡ Optimization: AdamW with custom LR")
print(f"📊 Training: 50 epochs with cosine LR")
print("=" * 60)
