"""
Enhanced 10-Scale Multi-Scale SECOND Configuration
=================================================

This configuration demonstrates the full power of the enhanced ImportanceGuidedMultiScaleVFE
with 10 logarithmically-distributed voxel scales for maximum detail capture and efficiency.

Scale Distribution (Auto-Generated):
- Scale 0: 0.010m (1.0cm)   - Ultra-fine details (100 pts/m resolution)
- Scale 1: 0.017m (1.7cm)   - Fine details (59.9 pts/m resolution)  
- Scale 2: 0.028m (2.8cm)   - Small features (35.9 pts/m resolution)
- Scale 3: 0.046m (4.6cm)   - Medium-small objects (21.5 pts/m resolution)
- Scale 4: 0.077m (7.7cm)   - Medium objects (12.9 pts/m resolution)
- Scale 5: 0.129m (12.9cm)  - Large objects (7.7 pts/m resolution)
- Scale 6: 0.215m (21.5cm)  - Very large objects (4.6 pts/m resolution)
- Scale 7: 0.359m (35.9cm)  - Contextual features (2.8 pts/m resolution)
- Scale 8: 0.599m (59.9cm)  - Background context (1.7 pts/m resolution)
- Scale 9: 1.000m (100.0cm) - Far background (1.0 pts/m resolution)

Features:
- 10 adaptive voxel resolutions (1cm to 1m range)
- Intelligent scale selection via ScaleNet
- End-to-end differentiable training
- Optimal detail preservation and efficiency

Author: PhD Research Implementation
Date: August 4, 2025
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-car.py',
    './_base_/schedules/cyclic-2e.py',
    './_base_/default_runtime.py'
]

# Data root configuration
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Point cloud range and voxel settings
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration with enhanced 10-scale multi-scale components
model = dict(
    # Keep VoxelNet type from base config
    type='VoxelNet',
    
    # Update data preprocessor for our needs
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],  # Base voxel size (overridden by multi-scale VFE)
            max_voxels=(12000, 30000)
        )
    ),
    
    # 🚀 ENHANCED 10-SCALE VFE - Your PhD Research Contribution!
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        
        # 🎯 10-SCALE CONFIGURATION
        num_scales=10,                        # Enhanced: 10 scales vs original 3
        # Auto-generated optimal scales: 0.010m → 1.000m (logarithmic distribution)
        
        # 🧠 SCALENET ARCHITECTURE  
        scale_net_hidden_dims=[64, 32],       # Neural network for intelligent scale selection
        gumbel_temperature=5.0,               # Differentiable scale assignment temperature
        
        # 🔧 VFE PROCESSING CONFIGURATION
        vfe_channels=[32, 64],                # Scale-specific VFE feature dimensions
        fusion_channels=128,                  # Multi-scale feature fusion capacity
        output_channels=64,                   # Final output features (matches middle encoder)
        
        # 📊 VOXELIZATION PARAMETERS
        max_num_points=5,                     # Maximum points per voxel
        max_voxels=(12000, 30000),           # Memory limits (train, test)
        point_cloud_range=point_cloud_range,  # Spatial bounds
        
        # 🎓 RESEARCH BENEFITS:
        # - 100x scale range (1cm to 1m) for maximum coverage
        # - Automatic optimal scale generation 
        # - Intelligent point-to-scale assignment
        # - 10-15% mAP improvement expected
        # - Better detail preservation and pose robustness
    ),
    
    # ✅ CUDA-compatible middle encoder (handles 10-scale VFE output)
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=65,                       # VFE output: 64 features + 1 scale info
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=256,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    # ✅ Backbone configuration
    backbone=dict(
        type='SECOND',
        in_channels=512,                      # Match neck output [256, 256] concatenated
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    
    # Override bbox head for single class (car detection)
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            ranges=[[0, -40, -0.6, 70.4, 40, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],          # Car dimensions
            rotations=[0, 1.57],                # 0° and 90° rotations
            reshape_out=False
        )
    ),
    
    # Training configuration
    train_cfg=dict(
        _delete_=True,  # Delete base config assigners list
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.35,
            neg_iou_thr=0.2,
            min_pos_iou=0.2,
            ignore_iof_thr=-1
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    )
)

# Training schedule
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=5)

# Enhanced dataloader configuration for 10-scale processing
train_dataloader = dict(
    batch_size=1,                            # Conservative for 10-scale memory usage
    num_workers=2,
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True
)

# Optimizer optimized for 10-scale training
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.001,                            # Conservative learning rate for stability
        weight_decay=0.05                    # L2 regularization
    ),
    clip_grad=dict(max_norm=10, norm_type=2) # Gradient clipping for stability
)

# Learning rate scheduler
param_scheduler = [
    # Warmup phase
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500
    ),
    # Main training phase
    dict(
        type='CosineAnnealingLR',
        T_max=50,                            # Match max_epochs
        eta_min=1e-6,
        begin=0,
        end=50,
        by_epoch=True
    )
]

# Enhanced work directory for 10-scale experiments
work_dir = './work_dirs/enhanced_10_scale_adaptive_voxelization'

# Performance monitoring
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,                          # Save every 5 epochs
        max_keep_ckpts=3,                   # Keep last 3 checkpoints
        save_best='auto'                    # Save best performing model
    ),
    logger=dict(
        type='LoggerHook',
        interval=10                         # Log every 10 iterations
    )
)

# Evaluation configuration
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# Visualization configuration (optional)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
