"""
Advanced Multi-Scale SECOND Configuration for Pedestrian Detection
================================================================

This configuration is specifically designed for pedestrian detection using
the adaptive multi-scale VFE. Pedestrians are smaller and more detailed
objects that benefit significantly from adaptive voxelization.

Features:
- Three finer voxel resolutions (0.025m, 0.05m, 0.1m) for pedestrians
- Separate VFE for each scale
- Attention-based importance weighting
- Learnable scale embeddings optimized for small objects
- Enhanced middle encoder

Author: PhD Research Implementation - Pedestrian Detection
Date: August 3, 2025
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-pedestrian.py',  # Pedestrian-only dataset
    './_base_/schedules/cyclic-2e.py',
    './_base_/default_runtime.py'
]

# Data root configuration
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Point cloud range and voxel settings
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration with our advanced multi-scale components optimized for pedestrians
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
            voxel_size=[0.05, 0.05, 0.1],  # Base voxel size
            max_voxels=(12000, 30000)
        )
    ),
    
    # Our advanced multi-scale VFE optimized for pedestrian detection
    # 🎓 YOUR PhD RESEARCH VFE - Adaptive Multi-Scale Voxelization for Pedestrians
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',  # Your research contribution!
        voxel_scales=[0.025, 0.05, 0.1],     # BALANCED: Fine detail + computational feasibility
        num_scales=3,
        scale_net_hidden_dims=[128, 64],      # ENHANCED: Larger ScaleNet for better learning
        gumbel_temperature=2.0,               # LOWERED: More decisive scale selection
        vfe_channels=[64, 128],               # ENHANCED: More VFE capacity
        fusion_channels=256,                  # ENHANCED: Better feature fusion
        output_channels=64,                   # Matches middle encoder input
        max_num_points=5,
        max_voxels=(16000, 40000),           # INCREASED: More voxels for detail
        point_cloud_range=point_cloud_range
    ),
    
    # ✅ CUDA-SAFE middle encoder (PhD research preserved in VFE above)
    middle_encoder=dict(
        type='SparseEncoder',  # Standard CUDA-compatible encoder
        in_channels=65,        # Match your research VFE output (64 + 1 scale info)
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=256,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    # ✅ FIXED: Update backbone input channels to match neck output (512 = 256 + 256)
    backbone=dict(
        type='SECOND',
        in_channels=512,  # CRITICAL: Match neck output [256, 256] concatenated
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    
    # Override bbox head for pedestrian detection
    bbox_head=dict(
        num_classes=1,  # Pedestrian only
        anchor_generator=dict(
            ranges=[[0, -40, -2.5, 70.4, 40, 1.0]],  # Adjusted for pedestrian height
            sizes=[[0.8, 0.6, 1.73]],  # Pedestrian size: width=0.8, depth=0.6, height=1.73
            rotations=[0, 1.57],  # Standing upright orientations
            reshape_out=False
        )
    ),
    
    # Single train configuration for pedestrian detection
    train_cfg=dict(
        _delete_=True,  # Delete base config assigners list
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.35,  # Suitable for small pedestrian objects
            neg_iou_thr=0.25,  # Adjusted for pedestrian detection
            min_pos_iou=0.25,  # Lower threshold for small objects
            ignore_iof_thr=-1
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    )
)

# Training configuration - FAST CONVERGENCE setup
train_cfg = dict(
    by_epoch=True, 
    max_epochs=20,     # SHORTER: Focus on rapid convergence validation
    val_interval=2     # MORE FREQUENT: Monitor convergence closely
)

# ✅ FIX: Proper dataloader configuration - set num_workers > 0 for persistent_workers
train_dataloader = dict(
    batch_size=1,  # Reduced batch size for memory efficiency
    num_workers=2,  # CRITICAL: Must be > 0 when persistent_workers=True
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,  # CRITICAL: Must be > 0 when persistent_workers=True
    persistent_workers=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,  # CRITICAL: Must be > 0 when persistent_workers=True
    persistent_workers=True
)

# Optimizer configuration - OPTIMIZED for faster convergence
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.002,           # INCREASED: Higher learning rate for faster convergence
        weight_decay=0.01,  # REDUCED: Less regularization for faster initial learning
        betas=(0.9, 0.999)  # Standard Adam betas
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # INCREASED: Allow larger gradients
)

# Learning rate scheduler - OPTIMIZED for faster convergence
param_scheduler = [
    # Aggressive warmup for stable fast learning
    dict(
        type='LinearLR',
        start_factor=0.1,       # START HIGHER: Faster warmup
        by_epoch=False,
        begin=0,
        end=200                 # SHORTER: Faster warmup period
    ),
    # Faster cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=20,              # SHORTER: Faster cycles for quicker convergence
        eta_min=5e-6,          # HIGHER: Don't go too low too fast
        begin=0,
        end=20,                # SHORTER: Focus on rapid initial learning
        by_epoch=True
    )
]

# Work directory for pedestrian detection experiments
work_dir = './work_dirs/pedestrian_adaptive_voxel_detection'
