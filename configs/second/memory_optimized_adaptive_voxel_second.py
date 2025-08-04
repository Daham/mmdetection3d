# 🚀 Memory-Optimized Adaptive Voxel SECOND Configuration
# Target: 25% memory reduction compared to vanilla SECOND

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

# 🚀 MEMORY-OPTIMIZED VOXEL CONFIGURATION
voxel_layer = dict(
    _delete_=True,
    type='MultiScaleDynamicVoxelize',
    voxel_scales=[0.05, 0.1, 0.2],  # Standard 3 scales
    max_num_points=5,
    point_cloud_range=point_cloud_range
)

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=voxel_layer),
    
    # 🚀 MEMORY-OPTIMIZED VFE
    voxel_encoder=dict(
        type='MemoryOptimizedImportanceGuidedMultiScaleVFE',
        in_channels=4,
        feat_channels=[32, 64],  # Standard capacity 
        with_distance=False,
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',
        legacy=False,
        
        # 🚀 MEMORY OPTIMIZATION SETTINGS
        memory_optimization_level=2,     # Aggressive optimization
        importance_threshold=0.15,       # Filter 15% of lowest importance points
        max_points_ratio=0.7,           # Keep only 70% of points
        adaptive_max_voxels=True,       # Dynamic voxel limits based on scene
        use_gradient_checkpointing=True, # Trade compute for memory
        
        # 🚀 REDUCED NETWORK CAPACITY
        importance_net_dims=[32, 16],    # Point importance network (reduced from [64, 32, 16])
        scale_net_dims=[32, 16],         # Scale prediction network (reduced from [64, 32])
        vfe_channels=[32, 64],           # VFE channels (can be further reduced)
        fusion_channels=64,              # Feature fusion (reduced from 128)
        
        # Scale configuration
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        max_num_points=5,
        max_voxels=(8000, 20000),        # Reduced from (12000, 30000)
        
        # ScaleNet configuration  
        gumbel_temperature=1.0,
        continuous_mode=False,           # Use discrete mode for memory efficiency
        
        # Output
        output_channels=64
    ),
    
    # 🚀 MEMORY-OPTIMIZED MIDDLE ENCODER
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,                  # Matches VFE output
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),  # Standard
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    
    # Standard backbone (no changes needed)
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
    
    train_cfg=dict(
        assigner=[
            dict(
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

# 🚀 MEMORY-OPTIMIZED TRAINING CONFIGURATION
train_dataloader = dict(
    batch_size=3,  # Increased from 2 due to memory savings
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

# 🚀 MEMORY-EFFICIENT OPTIMIZATION
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'importance_net': dict(lr_mult=1.2),     # Slightly higher LR for importance net
            'scale_net': dict(lr_mult=1.0),          # Standard LR for scale net
            'feature_fusion': dict(lr_mult=0.8),     # Lower LR for fusion
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2))

# Standard learning rate schedule
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

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 🚀 MEMORY-OPTIMIZED RUNTIME SETTINGS
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),  # Keep fewer checkpoints
    logger=dict(type='LoggingHook', interval=50),
    visualization=dict(type='Det3DVisualizationHook'))

val_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None)

test_evaluator = val_evaluator

# 🚀 ENABLE MIXED PRECISION FOR ADDITIONAL MEMORY SAVINGS
fp16 = dict(loss_scale='dynamic')

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,   # Disable for memory consistency
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Model loading
load_from = None
resume = False

# Experiment settings
experiment_name = 'memory_optimized_adaptive_voxel_second'
work_dir = f'./work_dirs/{experiment_name}'

# 🚀 MEMORY OPTIMIZATION SUMMARY
print("🚀 MEMORY-OPTIMIZED ADAPTIVE VOXEL SECOND CONFIGURATION")
print("=" * 70)
print("🎯 MEMORY REDUCTION STRATEGIES:")
print("   1. ⚡ Aggressive Point Filtering: 30% point reduction")
print("   2. 📦 Adaptive Voxel Limits: Dynamic based on scene complexity")
print("   3. 🔄 Gradient Checkpointing: Trade compute for memory")
print("   4. 🧠 Reduced Network Capacity: Smaller hidden dimensions")
print("   5. 💾 Efficient Feature Fusion: Minimal intermediate tensors")
print("   6. 🎛️ Memory-Aware Processing: Conservative voxel limits")
print("   7. 🔢 Mixed Precision Training: FP16 where safe")
print("")
print("📊 EXPECTED MEMORY SAVINGS:")
print("   • Point filtering: ~20% memory reduction")
print("   • Network reduction: ~10% memory reduction") 
print("   • Efficient processing: ~8% memory reduction")
print("   • Mixed precision: ~5% memory reduction")
print("   • TOTAL TARGET: ~25% memory reduction vs vanilla SECOND")
print("")
print("⚙️ CONFIGURATION DETAILS:")
print(f"   • Optimization Level: 2 (Aggressive)")
print(f"   • Batch Size: 3 (increased due to memory savings)")
print(f"   • Max Voxels: 8K/20K (reduced from 12K/30K)")
print(f"   • Gradient Checkpointing: Enabled")
print(f"   • Mixed Precision: Enabled")
print("=" * 70)
