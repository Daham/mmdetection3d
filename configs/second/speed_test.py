# SUPER FAST MINIMAL CONFIG - Just to test if training works at all
# This removes all potential bottlenecks to isolate the issue

_base_ = [
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Minimal dataset - just a few samples
train_dataloader = dict(
    batch_size=1,
    num_workers=0,  # No multiprocessing
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
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

# Extremely simplified model
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            voxel_size=[0.5, 0.5, 0.5],  # Same as vanilla
            max_voxels=(1000, 2000))),   # MUCH smaller than (16000, 40000)
    
    # Use standard VFE first to eliminate our module as the problem
    voxel_encoder=dict(type='HardSimpleVFE'),
    
    # Minimal middle encoder
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[8, 160, 140]),  # Much smaller grid
        
    # Minimal backbone
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[1, 1],  # Reduced layers
        layer_strides=[1, 2],
        out_channels=[64, 128]),
        
    # Minimal neck
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        out_channels=[128, 128]),
        
    # Minimal head
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=128,  # Reduced
        use_direction_classifier=False,  # Simplified
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)))

# Very short test
train_cfg = dict(max_epochs=1, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9))  # Simple optimizer

# Log every iteration
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# No validation to speed up
val_dataloader = None
val_evaluator = None
