# 🔬 PURE Adaptive Voxelization - True Research Implementation
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🔬 TRUE ADAPTIVE VOXELIZATION MODEL
model = dict(
    # REMOVE standard voxel preprocessor completely
    data_preprocessor=dict(
        _delete_=True,
        type='Det3DDataPreprocessor',
        # No voxel_layer - we'll do adaptive voxelization in the encoder
    ),
    
    # 🚀 PURE ADAPTIVE VOXEL ENCODER (processes raw points)
    voxel_encoder=dict(
        _delete_=True,
        type='PureAdaptiveVoxelLayer',
        point_cloud_range=point_cloud_range,
        base_voxel_size=[0.16, 0.16, 4.0],
        max_num_points=20,
        max_voxels=(8000, 20000),
        voxel_size_scale_range=(0.5, 2.0),
        importance_threshold=0.3,
    ),
    
    # Standard pipeline after voxelization
    middle_encoder=dict(
        _delete_=True,
        type='AdaptiveVoxelEncoder',
        in_channels=4,
        out_channels=64,
    ),
    
    # Proven architecture
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

# Conservative learning rate for stability
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.015,  # Start conservative with true adaptive
        betas=(0.9, 0.99), 
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Standard learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        by_epoch=False,
        begin=0,
        end=200,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=6,
        eta_min=0.002,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Standard training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Conservative batch size
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=1,
    persistent_workers=True,
    pin_memory=True
)

# Monitoring
default_hooks = dict(
    logger=dict(interval=25),
    checkpoint=dict(interval=1, save_best='auto', max_keep_ckpts=3),
)

work_dir = './work_dirs/adaptive_voxel_pure'

# 🎯 PURE ADAPTIVE RESEARCH GOALS:
# 1. Process raw point clouds (no pre-voxelization)
# 2. Learn optimal voxel sizes per region
# 3. Demonstrate memory efficiency gains
# 4. Show accuracy improvements over vanilla SECOND

# 📊 EXPECTED BEHAVIOR:
# - Important regions: Fine voxels (better accuracy)
# - Unimportant regions: Coarse voxels (memory efficiency)
# - Overall: Better accuracy + memory efficiency than vanilla
