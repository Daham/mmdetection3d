"""
ENHANCED Multi-Scale SECOND Configuration for Pedestrian Detection (10 Scales)
=============================================================================

🚀 ENHANCED: This configuration demonstrates the new 1-10 scales capability!

This configuration showcases the enhanced ImportanceGuidedMultiScaleVFE that now
supports up to 10 voxel scales for maximum detail capture and adaptive processing.

Key Enhancements:
- 🎯 10 automatically optimized voxel scales (0.01m to 1.0m range)
- 🔧 Logarithmic scale distribution for optimal coverage  
- 🧠 Enhanced ScaleNet with smart bias initialization
- ⚡ Backward compatible with existing 3-scale configs
- 🚀 Easy to configure: just set num_scales=10!

Author: Enhanced PhD Research Implementation - 10-Scale Pedestrian Detection
Date: August 4, 2025
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

# Model configuration with 🚀 ENHANCED 10-scale adaptive voxelization
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
            max_voxels=(16000, 40000)      # Increased for more detail
        )
    ),
    
    # 🚀 ENHANCED: 10-Scale Adaptive Multi-Scale VFE for Maximum Detail Capture
    # 🎓 YOUR PhD RESEARCH VFE - Now with 10 scales for ultimate pedestrian detection!
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',  # Your enhanced research contribution!
        
        # 🚀 KEY ENHANCEMENT: 10 automatically optimized scales
        num_scales=10,                        # 🎯 NEW: Just specify the number!
        # No need to manually specify voxel_scales - they're auto-generated optimally:
        # Scales will be: [0.010, 0.016, 0.025, 0.040, 0.063, 0.100, 0.158, 0.251, 0.398, 0.631]
        # Range: 1cm (fine details) to 63cm (context) - logarithmic distribution
        
        # Enhanced network architecture for 10 scales
        scale_net_hidden_dims=[128, 64, 32],  # Deeper ScaleNet for better 10-scale prediction
        gumbel_temperature=1.5,               # Lower temperature for more decisive selection
        
        # Enhanced VFE channels for richer feature extraction
        vfe_channels=[64, 128],               # More capacity for 10-scale processing
        fusion_channels=256,                  # Enhanced fusion for 10 scales
        output_channels=64,                   # Matches middle encoder input
        
        # Standard configuration
        max_num_points=5,
        max_voxels=(16000, 40000),           # Increased capacity
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

# Training configuration - Enhanced for 10-scale complexity
train_cfg = dict(
    by_epoch=True, 
    max_epochs=25,     # Slightly more epochs for 10-scale convergence
    val_interval=2     # Frequent validation
)

# ✅ FIX: Proper dataloader configuration - set num_workers > 0 for persistent_workers
train_dataloader = dict(
    batch_size=1,  # Keep small for 10-scale memory efficiency
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

# Optimizer configuration - Optimized for 10-scale training
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0015,          # Slightly lower LR for stable 10-scale training
        weight_decay=0.01,  # Light regularization
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # Gradient clipping for stability
)

# Learning rate scheduler - Optimized for 10-scale convergence
param_scheduler = [
    # Warmup for stable 10-scale learning
    dict(
        type='LinearLR',
        start_factor=0.1,       # Gentle start for complex 10-scale system
        by_epoch=False,
        begin=0,
        end=300                 # Longer warmup for 10 scales
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=25,              # Match max_epochs
        eta_min=1e-6,
        begin=0,
        end=25,
        by_epoch=True
    )
]

# Work directory for 10-scale pedestrian detection experiments
work_dir = './work_dirs/pedestrian_10_scale_adaptive_detection'

# 🎯 Expected Performance with 10 Scales:
# - Ultra-fine detail capture (1cm resolution)
# - Better pose variation handling
# - Improved distance-based performance  
# - Enhanced context understanding
# - Adaptive scale selection across full range
# 
# 📊 Scale Distribution (auto-generated):
# Scale 0: 0.010m (1.0cm) - Limb details, fine features
# Scale 1: 0.016m (1.6cm) - Joint articulation
# Scale 2: 0.025m (2.5cm) - Limb segments
# Scale 3: 0.040m (4.0cm) - Torso details
# Scale 4: 0.063m (6.3cm) - Body structure
# Scale 5: 0.100m (10.0cm) - Full pedestrian
# Scale 6: 0.158m (15.8cm) - Local context
# Scale 7: 0.251m (25.1cm) - Near environment
# Scale 8: 0.398m (39.8cm) - Spatial context
# Scale 9: 0.631m (63.1cm) - Global context
