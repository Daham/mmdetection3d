# 🔬 VANILLA SECOND BASELINE - For Memory Comparison
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🎯 VANILLA SECOND: Exact same setup as adaptive but no adaptive features
model = dict(
    # IDENTICAL voxel preprocessing as adaptive
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=5,
            voxel_size=[0.05, 0.05, 0.1],  # SAME as adaptive
            max_voxels=(12000, 30000)  # SAME as adaptive
        )
    ),
    
    # 🔬 VANILLA ENCODER: Standard HardSimpleVFE (no adaptivity)
    voxel_encoder=dict(
        _delete_=True,
        type='HardSimpleVFE',  # Vanilla SECOND encoder
    ),
    
    # IDENTICAL pipeline as adaptive
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    
    # IDENTICAL architecture as adaptive
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
    ),
    
    bbox_head=dict(
        type='Anchor3DHead',
        in_channels=384,
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

# IDENTICAL training setup as adaptive
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.015,
        betas=(0.9, 0.99), 
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# IDENTICAL schedule as adaptive
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

# Quick test for memory comparison
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader = dict(
    batch_size=2,  # SAME as adaptive
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
)

val_dataloader = dict(batch_size=1, num_workers=1)

default_hooks = dict(
    logger=dict(interval=1),  # Log every iteration for memory monitoring
    checkpoint=dict(interval=-1),  # No checkpoints for quick test
)

work_dir = './work_dirs/vanilla_second_memory_test'

# 🎯 VANILLA SECOND MEMORY BASELINE:
# This provides the exact memory usage baseline that adaptive 
# approaches should beat or at least match.
# 
# EXPECTED RESULTS:
# - Memory: ~500-600 MB (target for adaptive to beat)
# - Loss: Similar convergence pattern as adaptive
# - Speed: Baseline speed for comparison
