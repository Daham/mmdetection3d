"""
VALIDATION RUN: Baseline_02 - Fixed Multi-Scale (80 epochs)

Purpose: Test if fixed multi-scale (no learning) can improve over single-scale
Expected: May underperform baseline_01 if scale assignment is critical
"""

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.2]  # Base scale
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Fixed Multi-Scale (non-learnable)
model = dict(
    voxel_encoder=dict(
        type='SimpleFixedMultiScaleVFE',
        
        # Same scales as adaptive, but fixed assignment
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        
        # Match adaptive architecture
        vfe_channels=[32, 64],
        fusion_channels=64,
        output_channels=3,
        
        # Standard parameters
        max_num_points=5,
        max_voxels=(16000, 40000),
        point_cloud_range=point_cloud_range,
        
        # Fixed uniform assignment strategy
        assignment_strategy='uniform',
        
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    ),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,  # 3 features + 1 scale info
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)))

# Optimizer (same as baseline_01)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.001,
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate schedule with warmup
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
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
    val_interval=5,
    dynamic_intervals=[(75, 1)]
)

# Data loading (smaller batch due to memory)
train_dataloader = dict(
    batch_size=4,  # Reduced for multi-scale memory
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
        interval=10,
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
