_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.2]  # Base scale for multi-scale (matching paper)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Fixed Multi-Scale Baseline with Gumbel-Softmax Weighted Fusion
# Research Focus: Isolates multi-scale benefits from learnable scale selection
model = dict(
    voxel_encoder=dict(
        type='FixedMultiScaleVFE',
        voxel_scales=[0.05, 0.1, 0.2],  # Fixed scales exactly as described in paper
        max_num_points=3,  # Reduced from 5 to save memory
        max_voxels=(8000, 20000),  # Reduced from (12000, 30000) to save memory
        point_cloud_range=point_cloud_range,
        vfe_channels=[16, 32],  # Reduced from [32, 64] to save memory
        fusion_channels=64,  # Reduced from 128 to save memory
        output_channels=32,  # Reduced from 64 to save memory
        with_distance=True,
        with_cluster_center=True,
        with_voxel_center=True,
        # Gumbel-Softmax fusion parameters
        use_gumbel_fusion=True,  # Enable learnable weighted fusion
        gumbel_temperature=2.0,  # Initial temperature
        temperature_decay=0.995,  # Decay rate
        min_temperature=0.5      # Minimum temperature
    ),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=32,  # Match our reduced VFE output channels
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
        max_epochs=5,
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

# Enhanced optimizer configuration for Gumbel-Softmax training
optim_wrapper = dict(
    type='OptimWrapper',  # Use standard optimizer (no mixed precision for now)
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),  # Reduced LR due to smaller batch
    clip_grad=dict(max_norm=10, norm_type=2),
    # Special learning rates for Gumbel-Softmax components
    paramwise_cfg=dict(
        custom_keys={
            'temperature': dict(lr_mult=0.1),  # Lower learning rate for temperature
            'scale_weight_net': dict(lr_mult=0.5),  # Moderate learning for scale predictor
        })
)

# Memory-efficient training settings
train_dataloader = dict(
    batch_size=2,  # Reduced batch size for memory efficiency
    num_workers=2,  # Reduced workers
    persistent_workers=True
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
    max_epochs=3,  # Reduced epochs for testing
    val_interval=1
)

default_hooks = dict(
    logger=dict(interval=50, type='LoggerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook'),
)
