_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.2]  # Base scale for multi-scale (matching paper)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Fixed Multi-Scale Baseline using new FixedMultiScaleVFEBaseline
# Research Focus: Same multi-scale architecture but with fixed (non-learnable) scale assignment
model = dict(
    voxel_encoder=dict(
        type='SimpleFixedMultiScaleVFE',  # Use new fixed multi-scale implementation
        
        # Multi-scale configuration (SAME as adaptive)
        voxel_scales=[0.05, 0.1, 0.2],  # Fixed scales (non-learnable)
        num_scales=3,
        
        # Match adaptive config exactly
        vfe_channels=[32, 64],  # Same as adaptive
        fusion_channels=64,     # Same as adaptive
        output_channels=3,      # Same as adaptive
        
        # Standard VFE parameters (SAME as adaptive)
        max_num_points=5,
        max_voxels=(16000, 40000),
        point_cloud_range=point_cloud_range,
        
        # Fixed assignment strategy
        assignment_strategy='uniform',  # Options: 'uniform', 'round_robin', 'distance_based'
        
        # Standard parameters
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    ),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,  # Match adaptive config (3+1 scale info)
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),  # Standard progression
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    train_cfg=dict(
        _delete_=True,
        max_epochs=2,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False))

# Enhanced optimizer configuration (Match proven working config)
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Re-enable mixed precision (HardVFE supports it)
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),  # Match adaptive config
    clip_grad=dict(max_norm=10, norm_type=2),
)

# Memory-efficient training settings (Match other baselines)
train_dataloader = dict(
    batch_size=4,  # Match other baselines for fair comparison
    num_workers=2,  # Standard workers
    persistent_workers=True  # Enable for efficiency
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True
)

# Override to use epoch-based training
train_cfg = dict(
    _delete_=True,  # Delete the base config
    type='EpochBasedTrainLoop',
    max_epochs=2,  # Match other baselines for fair comparison
    val_interval=1
)

default_hooks = dict(
    logger=dict(interval=50, type='LoggerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook'),
)
