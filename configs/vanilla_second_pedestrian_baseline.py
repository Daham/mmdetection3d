"""
Vanilla SECOND Baseline Configuration for Pedestrian Detection Benchmarking
==========================================================================

This configuration serves as a baseline to benchmark against the adaptive 
multi-scale VFE for pedestrian detection. Uses standard SECOND components 
with IDENTICAL settings to the adaptive config except for the VFE module.

Pedestrian Detection Focus:
- Smaller objects (0.8 × 0.6 × 1.73m) vs cars (3.9 × 1.6 × 1.56m)
- Requires finer resolution for accurate detection
- More challenging for fixed voxelization approaches
- Perfect test case for adaptive voxelization benefits

Author: PhD Research Benchmarking - Pedestrian Detection
Date: August 3, 2025
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-pedestrian.py',  # Pedestrian-only dataset
    './_base_/schedules/cyclic-2e.py',
    './_base_/default_runtime.py'
]

# Data root configuration (IDENTICAL to adaptive config)
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Point cloud range and voxel settings (IDENTICAL to adaptive config)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration with VANILLA SECOND components for pedestrian detection
model = dict(
    # Keep VoxelNet type from base config
    type='VoxelNet',
    
    # Update data preprocessor (IDENTICAL to adaptive config)
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],  # Fixed voxel size (no adaptation) - may miss fine pedestrian details
            max_voxels=(12000, 30000)
        )
    ),
    
    # 📊 BASELINE: Standard SECOND VFE (no adaptive voxelization)
    # This is the KEY DIFFERENCE from the adaptive config
    # Challenge: Fixed 0.05m voxels may be too coarse for pedestrian features
    voxel_encoder=dict(
        type='HardSimpleVFE',  # Standard vanilla VFE
        num_features=4,        # x, y, z, intensity
    ),
    
    # ✅ CUDA-SAFE middle encoder (IDENTICAL to adaptive config)
    middle_encoder=dict(
        type='SparseEncoder',  # Standard CUDA-compatible encoder
        in_channels=4,         # HardSimpleVFE outputs 4 channels (vs 65 for adaptive)
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=256,
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    
    # ✅ Update backbone input channels (IDENTICAL to adaptive config)
    backbone=dict(
        type='SECOND',
        in_channels=512,  # Match neck output [256, 256] concatenated
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    
    # Override bbox head for pedestrian detection (IDENTICAL to adaptive config)
    bbox_head=dict(
        num_classes=1,  # Pedestrian only
        anchor_generator=dict(
            ranges=[[0, -40, -2.5, 70.4, 40, 1.0]],  # Adjusted for pedestrian height
            sizes=[[0.8, 0.6, 1.73]],  # Pedestrian size: width=0.8, depth=0.6, height=1.73
            rotations=[0, 1.57],  # Standing upright orientations
            reshape_out=False
        )
    ),
    
    # Single train configuration for pedestrian detection (IDENTICAL to adaptive config)
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

# Training configuration (IDENTICAL to adaptive config)
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=5)

# ✅ Dataloader configuration (IDENTICAL to adaptive config)
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

# Optimizer configuration (IDENTICAL to adaptive config)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate scheduler (IDENTICAL to adaptive config)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        eta_min=1e-6,
        begin=0,
        end=40,
        by_epoch=True
    )
]

# Work directory for pedestrian baseline experiments
work_dir = './work_dirs/vanilla_second_pedestrian_baseline'
