"""
VALIDATION RUN: Baseline_01 - Single-Scale HardVFE (80 epochs)

Purpose: Establish solid single-scale SECOND baseline performance
Expected: 71-73% 3D AP@0.7 (standard KITTI Car performance)
"""

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.2]  # Standard SECOND voxel size
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Standard SECOND with HardSimpleVFE
model = dict(
    voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=4,  # x, y, z, intensity
    ),
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],  # Car dimensions
            rotations=[0, 1.57],
            reshape_out=True)))

# Stable optimizer configuration
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.001,           # Standard LR for SECOND
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate schedule with warmup for stability
param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=75,
        by_epoch=True,
        begin=5,
        end=80,
        convert_to_iter_based=True
    )
]

# Training configuration
train_cfg = dict(
    max_epochs=80,
    val_interval=5,  # Validate every 5 epochs
    dynamic_intervals=[(75, 1)]  # More frequent validation near end
)

# Data loading
train_dataloader = dict(
    batch_size=6,  # Standard batch size for SECOND
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = val_dataloader

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=10,  # Save every 10 epochs
        max_keep_ckpts=3,
        save_best='KITTI/Car_3d_moderate_strict',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Evaluation
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    pcd_limit_range=point_cloud_range
)

test_evaluator = val_evaluator

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

log_level = 'INFO'
load_from = None
resume = False
