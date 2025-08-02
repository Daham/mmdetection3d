# ULTRA-MINIMAL CONFIG FOR TESTING
# This removes all potential bottlenecks

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
]

# Minimal data loading
train_dataloader = dict(
    batch_size=1,
    num_workers=0,  # No multiprocessing
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),  # No shuffling
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_train.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        test_mode=False))

# Simple model
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=5,  # Reduced from 32
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=[0.05, 0.05, 0.1],
        max_voxels=(1000, 2000)),  # Reduced from (16000, 40000)
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408]),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[1, 1],  # Reduced from [5, 5]
        layer_strides=[1, 2],
        out_channels=[64, 128]),  # Reduced from [128, 256]
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],  # Matched to backbone
        upsample_strides=[1, 2],
        out_channels=[128, 128]),  # Reduced
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True))

# Very short training
train_cfg = dict(max_epochs=1, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9))  # Simpler optimizer

# Log every single step
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# Disable validation to speed up
val_dataloader = None
val_evaluator = None
