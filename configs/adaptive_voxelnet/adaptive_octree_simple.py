"""SIMPLIFIED Adaptive Octree Configuration - Size as Feature

Key Innovation: Instead of complex adaptive backbone, we encode voxel size 
as an additional feature channel. SECOND's sparse convolutions learn to 
use this size information naturally!

Architecture:
1. Octree VFE: Variable voxels → features [N, C] + sizes [N, 1]
2. Concatenate: [N, C+1] where last channel = voxel size
3. Standard SECOND: Sparse conv learns size-aware features
4. Standard FPN + Detection Head

This is MUCH simpler and more memory-efficient!
"""

custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

_base_ = [
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]  # Base grid for sparse conv

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False,
        voxel_layer=None
    ),
    
    # Octree VFE with size encoding
    voxel_encoder=dict(
        type='AdaptiveOctreeVFE',
        in_channels=4,  # x, y, z, intensity
        feat_channels=[64, 128, 256],
        max_depth=6,  # Variable voxel sizes
        min_points_per_voxel=5,
        max_points_per_voxel=100,
        learnable_split=True,
        split_temperature=1.0,
        point_cloud_range=point_cloud_range,
        # NEW: Output size as feature
        include_size_in_features=True  # Concatenate size to features
    ),
    
    # Simple middle encoder: just maps to grid
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=257,  # 256 from VFE + 1 for size!
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    # Standard SECOND backbone
    backbone=dict(
        type='SECOND',
        in_channels=128,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    
    # Standard neck
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]
    ),
    
    # Detection head
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,  # 256 + 256 from neck
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],
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
    
    train_cfg=dict(
        assigner=[dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1
        )],
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

# Smaller batch size for safety
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True
    )
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=24)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)
)
