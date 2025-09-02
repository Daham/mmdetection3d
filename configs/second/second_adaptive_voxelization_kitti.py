_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.2]  # Base scale for adaptive voxelization (matching paper)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Override the voxel encoder to use memory-optimized importance-guided multi-scale VFE
model = dict(
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',  # Use full version instead of memory-optimized
        
        # Multi-scale configuration
        voxel_scales=[0.05, 0.1, 0.2],  # Research scales
        num_scales=3,
        
        # CRITICAL FIX: Proper output channels (3+1=4 to match SparseEncoder)
        output_channels=3,  # Will become 4 with +1 scale info
        vfe_channels=[32, 64],  # Good internal capacity
        fusion_channels=64,  # Reasonable fusion
        
        # Standard VFE parameters
        max_num_points=5,
        max_voxels=(16000, 40000),
        point_cloud_range=point_cloud_range,
        
        # FIXED: Stable adaptive parameters
        gumbel_temperature=0.5,  # Lower temperature for stable learning
        continuous_mode=False,  # Discrete for stability
        
        # Disable aggressive optimizations for fair comparison
        # memory_optimization_level=0,  # Full capacity
        # use_gradient_checkpointing=False,  # No checkpointing
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

# FIXED: Lower learning rate for complex adaptive components
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed precision training
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),  # Reduced from 0.003 for stability
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Override to use epoch-based training (matching base config)
train_cfg = dict(
    _delete_=True,  # Delete the base config
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1
)

default_hooks = dict(
    logger=dict(interval=50, type='LoggerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook'),
)
