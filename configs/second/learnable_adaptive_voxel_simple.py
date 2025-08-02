# WORKING Adaptive Voxelization Configuration - Simple Start
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Basic configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🚀 MINIMAL ADAPTIVE VOXELIZATION MODEL
model = dict(
    # Keep standard voxel layer in data_preprocessor for compatibility
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=5,
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000)
        )
    ),
    
    # Replace voxel encoder with your adaptive version
    voxel_encoder=dict(
        _delete_=True,
        type='AdaptiveLearnableVoxelLayer',
        point_cloud_range=point_cloud_range,
        base_voxel_size=[0.16, 0.16, 4.0],
        max_num_points=20,
        max_voxels=(8000, 20000),
        voxel_size_scale_range=(0.5, 2.0),
        importance_threshold=0.5,
    ),
    
    # Replace middle encoder with your adaptive version
    middle_encoder=dict(
        _delete_=True,
        type='AdaptiveVoxelEncoder',
        in_channels=4,
        out_channels=64,
    ),
    
    # Standard components - keep simple
    backbone=dict(
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[32, 64, 128],
    ),
    
    neck=dict(
        in_channels=[32, 64, 128],
        upsample_strides=[1, 2, 4],
        out_channels=[64, 64, 64],
    ),
    
    # Simplified detection head
    bbox_head=dict(
        in_channels=192,
        feat_channels=192,
        num_classes=1,
    ),
    
    # Keep standard training config
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
)

# Use standard KITTI dataset - no custom filtering for now
# This ensures we have actual data to train on

# Simple training settings
train_cfg = dict(max_epochs=2, val_interval=1)  # Start with just 2 epochs

# Reduced batch size and simplified optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.001),  # Lower learning rate
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Simplified hooks
default_hooks = dict(
    logger=dict(interval=10),  # Log more frequently for debugging
    checkpoint=dict(interval=1),
)

work_dir = './work_dirs/adaptive_voxel_simple'
