_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-cyclist-only.py',  # Use cyclist-only dataset
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Adaptive multi-scale voxelization for Cyclist class (matching Car config)
model = dict(
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',  # Use the correct registered class
        
        # Multi-scale configuration
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        
        # Output channels configuration
        output_channels=3,
        vfe_channels=[32, 64],
        fusion_channels=64,
        
        # Standard VFE parameters
        max_num_points=5,
        max_voxels=(16000, 40000),
        point_cloud_range=point_cloud_range,
        
        # Adaptive parameters
        gumbel_temperature=0.5,
        continuous_mode=False,
    ),
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[1.76, 0.6, 1.73]],  # Cyclist dimensions
            rotations=[0, 1.57],
            reshape_out=True)),
    train_cfg=dict(
        _delete_=True,
        max_epochs=5,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=5,
    val_interval=1
)

default_hooks = dict(
    logger=dict(interval=50, type='LoggerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook'),
)
