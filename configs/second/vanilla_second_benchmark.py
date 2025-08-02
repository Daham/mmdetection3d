# 🔬 VANILLA SECOND BENCHMARK - Exact Match for Comparison
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration - IDENTICAL to adaptive config
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🎯 VANILLA SECOND: Standard configuration exactly matching adaptive config
model = dict(
    # IDENTICAL voxel preprocessing
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=5,
            voxel_size=[0.05, 0.05, 0.1],  # IDENTICAL voxel size
            max_voxels=(12000, 30000)      # IDENTICAL voxel count
        )
    ),
    
    # 🔬 STANDARD VOXEL ENCODER: Simple mean aggregation (no adaptive features)
    voxel_encoder=dict(
        type='HardSimpleVFE',  # Standard simple voxel feature extractor
        num_features=4,
    ),
    
    # IDENTICAL middle encoder configuration  
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,  # Standard input from HardSimpleVFE
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    
    # IDENTICAL backbone configuration (adjusted for standard SparseEncoder output)
    backbone=dict(
        type='SECOND',
        in_channels=256,  # Accept SparseEncoder default output
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    
    # IDENTICAL neck configuration
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],  # Match backbone out_channels
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],  # Standard configuration
    ),
    
    # IDENTICAL bbox head configuration
    bbox_head=dict(
        type='Anchor3DHead',
        in_channels=384,  # 128 * 3 from neck outputs
        feat_channels=384,
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
    
    # IDENTICAL training configuration
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

# IDENTICAL learning rate and optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.015,  # IDENTICAL learning rate
        betas=(0.9, 0.99), 
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# IDENTICAL schedule
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

# IDENTICAL training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# IDENTICAL data loading configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
)

val_dataloader = dict(batch_size=1, num_workers=1)

# IDENTICAL logging and checkpointing
default_hooks = dict(
    logger=dict(interval=25),
    checkpoint=dict(interval=1, save_best='auto', max_keep_ckpts=3),
)

work_dir = './work_dirs/vanilla_second_benchmark'

# 🎯 VANILLA SECOND BASELINE:
# 1. Standard HardSimpleVFE voxel encoder (simple mean aggregation)
# 2. Standard SparseEncoder middle encoder
# 3. Identical training setup for fair comparison
# 4. Baseline performance for adaptive approach evaluation
