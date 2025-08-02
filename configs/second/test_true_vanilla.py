# TEST CONFIG - Validate that our module is identical to vanilla

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# IDENTICAL settings to vanilla SECOND
voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    
    # Use actual vanilla HardSimpleVFE to verify data processing is identical
    voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    
    bbox_head=dict(num_classes=1))

# IDENTICAL training settings to vanilla SECOND
train_cfg = dict(max_epochs=80, val_interval=10)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))

# Standard logging
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)
