"""
VALIDATION RUN: Baseline_03 - Adaptive Learnable Multi-Scale (80 epochs)

Purpose: Test # Training configuration - STABILITY: Longer warmup
train_cfg = dict(
    max_epochs=80,
    val_interval=5,
    dynamic_intervals=[(75, 1)]
)

# Data loading adaptive scale selection outperforms fixed approaches
Expected: Should outperform both baseline_01 and baseline_02 by 3-5%
Key Innovation: Importance-guided scale assignment with Gumbel-Softmax
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

# Adaptive Multi-Scale with Importance-Guided Selection
model = dict(
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        
        # Learnable multi-scale configuration
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        
        # Architecture (matching fixed multi-scale)
        output_channels=3,
        vfe_channels=[32, 64],
        fusion_channels=64,
        
        # Standard parameters
        max_num_points=5,
        max_voxels=(16000, 40000),
        point_cloud_range=point_cloud_range,
        
        # Adaptive learning parameters
        gumbel_temperature=2.0,  # Start high for exploration
        temperature_decay=0.995,  # Gradual annealing
        min_temperature=0.5,      # End low for exploitation
        continuous_mode=False,    # Discrete for stability
        
        # Importance scoring
        importance_threshold=0.1,
        use_importance_weighting=True,
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

# STABILITY FIX: Lower LR for complex adaptive components
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0005,  # REDUCED from 0.001 for stability
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# STABILITY FIX: Longer warmup for adaptive learning
param_scheduler = [
    # Longer warmup (10 epochs instead of 5)
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=10,  # Extended warmup
        convert_to_iter_based=True
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=70,
        by_epoch=True,
        begin=10,
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

# Data loading
train_dataloader = dict(
    batch_size=4,  # Reduced for multi-scale + adaptive learning
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

# Hooks with EMA for stability
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
    visualization=dict(type='Det3DVisualizationHook'),
    # EMA for stable evaluation
    ema=dict(
        type='EMAHook',
        momentum=0.0002,
        priority='ABOVE_NORMAL',
        strict_load=False
    )
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
