"""Configuration for naive multi-scale baseline (should perform poorly).

This config uses three fixed voxel sizes [0.05, 0.1, 0.2]m but with
NO learning - just uniform/random assignment. This demonstrates that
simply using multiple scales without intelligent assignment fails.
"""

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

# Multi-scale with fixed assignment (no learning)
voxel_size = [0.05, 0.1, 0.2]  # Three fixed scales
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.1, 0.1, 0.2],  # Use medium scale for preprocessing
            max_voxels=(16000, 40000)
        )
    ),
    voxel_encoder=dict(
        type='MultiScaleVFE',  # Process at multiple scales
        voxel_sizes=[0.05, 0.1, 0.2],
        num_features=4,
        point_cloud_range=point_cloud_range,
        learned_assignment=False,  # NO LEARNING - uniform assignment
        fusion_method='concat'  # Simple concatenation
    ),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=12,  # 4 features × 3 scales
        sparse_shape=[41, 800, 704],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    backbone=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 128, 128],
        out_channels=128,
        num_outs=3
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=384,
        feat_channels=384,
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
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
    ),
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1
            )
        ],
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

# Training settings
train_dataloader = dict(batch_size=4, num_workers=4)  # Reduced batch size (3 scales)
val_dataloader = dict(batch_size=1, num_workers=1)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.003),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-5,
        begin=0,
        end=20
    )
]

# Runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)
)

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Work directory
work_dir = './work_dirs/multi_scale_fixed'

# Expected performance: ~41-45% (WORSE than single-scale!)
# This demonstrates that naive multi-scale without learning FAILS
