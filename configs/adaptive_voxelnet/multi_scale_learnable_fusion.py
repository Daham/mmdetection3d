"""Configuration for learnable multi-scale fusion (your previous work).

This uses fixed scales [0.05, 0.1, 0.2]m with Gumbel-Softmax to learn
which scale to use per voxel. This is your previous work that achieved
~68% AP - better than naive multi-scale but still uses fixed voxel sizes.
"""

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

# Multi-scale with learned assignment (your previous work)
voxel_sizes = [0.05, 0.1, 0.2]  # Three fixed scales
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.1, 0.1, 0.2],
            max_voxels=(16000, 40000)
        )
    ),
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',  # Your previous work
        in_channels=4,
        feat_channels=[64, 128],
        voxel_sizes=voxel_sizes,
        point_cloud_range=point_cloud_range,
        # Learnable fusion parameters
        use_gumbel_softmax=True,
        gumbel_tau=1.0,
        tau_decay=0.99,
        importance_threshold=0.3,
        # Skip connections and importance filtering
        use_skip_connection=True,
        use_importance_filter=True,
        dropout_rate=0.1
    ),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=128,  # Output from fusion
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
train_dataloader = dict(batch_size=5, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)

# Optimizer with weight decay
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Learning rate schedule with warmup
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
        eta_min=1e-5,
        begin=0,
        end=20,
        by_epoch=True
    )
]

# Runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

# Custom hooks for Gumbel temperature decay
custom_hooks = [
    dict(
        type='GumbelTauDecayHook',
        decay_rate=0.99,
        min_tau=0.5
    )
]

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Work directory
work_dir = './work_dirs/multi_scale_learnable_fusion'

# Expected performance: ~68-70% (your previous work)
# Better than naive multi-scale because of learned assignment
# But still limited by fixed voxel sizes
