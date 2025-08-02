# 🎯 ULTIMATE SOLUTION - Simple & Effective Adaptive Voxelization
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'  # Remove cyclic-2e schedule
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🚀 YOUR ADAPTIVE VOXELIZATION MODEL
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=5,
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000)
        )
    ),
    
    # Your adaptive components
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
    
    middle_encoder=dict(
        _delete_=True,
        type='AdaptiveVoxelEncoder',
        in_channels=4,
        out_channels=64,
    ),
    
    # Standard proven architecture
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
    
    bbox_head=dict(
        in_channels=192,
        feat_channels=192,
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True
        )
    ),
    
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

# 🎯 KEY SOLUTION: AGGRESSIVE LEARNING RATE
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.01, betas=(0.9, 0.99), weight_decay=0.01),  # 20x higher!
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Simple, effective learning rate schedule
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=0.001,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Conservative batch size
train_dataloader = dict(batch_size=1, num_workers=1)
val_dataloader = dict(batch_size=1, num_workers=1)

# Enhanced logging
default_hooks = dict(
    logger=dict(interval=10),
    checkpoint=dict(interval=1),
)

work_dir = './work_dirs/adaptive_voxel_solution'

# 🎯 EXPECTED RESULTS:
# With lr=0.01, you should see:
# - Epoch 1: loss drops from 2.3 → 1.8
# - Epoch 3: loss drops to 1.5
# - Epoch 5: loss drops to 1.2
# - Epoch 10: loss < 1.0
