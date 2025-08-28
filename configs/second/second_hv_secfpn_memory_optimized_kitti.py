_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Override the voxel encoder to use memory-optimized importance-guided multi-scale VFE
model = dict(
    voxel_encoder=dict(
        type='MemoryOptimizedImportanceGuidedMultiScaleVFE',
        in_channels=4,
        output_channels=3,  # Will become 4 with +1 for scale info, matching SparseEncoder expectation
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        memory_optimization_level=2,  # Level 2: aggressive optimization
        use_gradient_checkpointing=True,  # Memory optimization
        importance_threshold=0.1,
        vfe_channels=[16, 32],  # Reduced channels for memory efficiency
        fusion_channels=32,     # Reduced from default 64
    ),
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

# Optimizer and training configuration
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed precision training
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Override to use epoch-based training (matching base config)
train_cfg = dict(
    _delete_=True,  # Delete the base config
    type='EpochBasedTrainLoop',
    max_epochs=5,
    val_interval=1
)

default_hooks = dict(
    logger=dict(interval=50, type='LoggerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook'),
)
