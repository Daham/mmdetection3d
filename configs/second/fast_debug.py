# FAST DEBUG CONFIG - Minimal dataset for testing
# This will help identify if data loading is the bottleneck

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Minimal dataset configuration for debugging
train_dataloader = dict(
    batch_size=1,  # Very small batch
    num_workers=0,  # No multiprocessing - easier to debug
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=False,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='LiDAR'))

# Standard model
model = dict(
    voxel_encoder=dict(type='HardSimpleVFE'),
    bbox_head=dict(num_classes=1))

# Very short test
train_cfg = dict(max_epochs=1, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))

# Log every iteration to see progress
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1)
)
