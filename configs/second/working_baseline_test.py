# 🔧 SIMPLIFIED WORKING BASELINE TEST
# Let's start with a WORKING configuration and gradually add adaptive features

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🔧 SIMPLE WORKING MODEL: Standard VoxelNet with basic VFE
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000)
        )
    ),
    
    # 🔧 SIMPLE VFE: Start with working baseline
    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.05, 0.05, 0.1],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),
    
    # 🔧 WORKING MIDDLE ENCODER
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128
    ),
    
    # 🔧 WORKING BACKBONE
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    # 🔧 WORKING NECK
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    
    # 🔧 WORKING BBOX HEAD
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0/9.0,
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2
        )
    ),
    
    # Training configuration
    train_cfg=dict(
        _delete_=True,  # Clear base config
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
    
    # Test configuration
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50
    )
)

# 🔧 CONSERVATIVE OPTIMIZER: Ensure training works
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# 🔧 SIMPLE SCHEDULE
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=32,
        eta_min_ratio=1e-4,
        begin=500,
        end=36
    )
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)  # SHORT test
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Data configuration
train_dataloader = dict(
    batch_size=2,  # Small batch for stability
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type='RepeatDataset',
        times=1,  # No repeat for test
        dataset=dict(type='KittiDataset')
    )
)
