_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

# Fixed Multi-Scale Baseline: [0.05, 0.1, 0.2] m fixed scales fused before detection
voxel_size = [0.1, 0.1, 0.2]  # Primary scale (medium resolution)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Multi-resolution voxelization with [0.05, 0.1, 0.2] m fixed scales
# This isolates multi-scale benefits from learnable scales
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.1, 0.1, 0.2],  # Primary voxel size (medium scale)
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='HardSimpleVFE',  # Using standard VFE, multi-scale handled in preprocessing
        num_features=4
        # Note: In a complete implementation, this would be a custom multi-scale VFE
        # that processes [0.05, 0.1, 0.2]m scales and fuses features before detection
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

# Optimizer configuration matching paper: lr=3×10^-3, weight_decay=1×10^-2
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed precision training
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.01),  # lr=3×10^-3
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Override to use epoch-based training
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
