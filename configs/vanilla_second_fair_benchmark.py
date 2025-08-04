"""
Vanilla SECOND Baseline Configuration for Fair Benchmarking
==========================================================

This configuration serves as a baseline to benchmark against the adaptive 
multi-scale VFE. Uses standard SECOND components with IDENTICAL settings
to the adaptive config except for the VFE module.

Fair Comparison Setup:
- Vanilla HardSimpleVFE vs ImportanceGuidedMultiScaleVFE
- All other components IDENTICAL for fair comparison
- Same data, optimizer, scheduler, training settings, and architecture
- Only difference: adaptive voxelization vs fixed voxelization

Author: PhD Research Benchmarking
Date: August 3, 2025
"""

_base_ = [
    './_base_/models/second_hv_secfpn_kitti.py',
    './_base_/datasets/kitti-3d-car.py',
    './_base_/schedules/cyclic-2e.py',
    './_base_/default_runtime.py'
]

# Data root configuration (IDENTICAL to adaptive config)
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Point cloud range and voxel settings (IDENTICAL to adaptive config)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Model configuration with VANILLA SECOND components
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
            voxel_size=[0.05, 0.05, 0.1],  # Fixed voxel size (no adaptation)
            max_voxels=(12000, 30000)
        )
    ),
    
    # 📊 BASELINE: Standard SECOND VFE (no adaptive voxelization)
    # This is the KEY DIFFERENCE from the adaptive config
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
    
    # Override bbox head for single class (IDENTICAL to adaptive config)
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            ranges=[[0, -40, -0.6, 70.4, 40, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False
        )
    ),
    
    # Single train configuration (IDENTICAL to adaptive config)
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

# Work directory for baseline experiments
work_dir = './work_dirs/vanilla_second_fair_benchmark'
