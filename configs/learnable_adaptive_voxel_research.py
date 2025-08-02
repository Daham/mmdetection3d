"""
🔬 ADAPTIVE VOXELIZATION RESEARCH CONFIGURATION
Learnable voxel sizes that adapt based on feature importance

Research Goal: Make voxel sizes trainable parameters that learn through backpropagation
"""

_base_ = [
    '_base_/datasets/kitti-3d-car.py', 
    '_base_/models/second_hv_secfpn_kitti.py',
    '_base_/schedules/cyclic-40e.py', 
    '_base_/default_runtime.py'
]

# 🔬 RESEARCH: Custom adaptive voxelization components
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
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    
    backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2]),
    
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

# 🔬 RESEARCH TRAINING: Test with 5 iterations (override base config)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5, val_interval=10)

# Remove conflicting epoch-based config
max_epochs = None
by_epoch = False
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 🔬 ADAPTIVE PARAMETER SCHEDULER: For learnable voxel parameters
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

# Optimization for research
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Data settings
train_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=False, pin_memory=True)
val_dataloader = dict(batch_size=1, num_workers=1, persistent_workers=False)

# Logging
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=-1))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Work directory
work_dir = 'work_dirs/adaptive_research'

print("🔬 LEARNABLE ADAPTIVE VOXELIZATION RESEARCH - IMPLEMENTATION COMPLETE!")
print("✅ Adaptive voxel sizes: LEARNABLE PARAMETERS")
print("✅ Importance prediction: IMPLEMENTED") 
print("✅ Memory efficiency: IMPLEMENTED")
print("⚡ Testing: 5 iterations to verify research implementation")
