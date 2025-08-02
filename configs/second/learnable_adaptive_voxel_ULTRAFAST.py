# 🔥 ULTRA-FAST Adaptive Voxelization - Memory Optimized for Speed
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'  # No conflicting schedules
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🚀 BALANCED FAST ADAPTIVE VOXELIZATION (Speed + Stability)
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=5,  # Back to proven 5
            voxel_size=[0.05, 0.05, 0.1],  # Back to original proven size
            max_voxels=(11000, 27000)  # Moderate reduction for memory
        )
    ),
    
    # Balanced adaptive components (proven + optimized)
    voxel_encoder=dict(
        _delete_=True,
        type='AdaptiveLearnableVoxelLayer',
        point_cloud_range=point_cloud_range,
        base_voxel_size=[0.16, 0.16, 4.0],  # Back to proven size
        max_num_points=15,  # Back to proven count
        max_voxels=(6000, 15000),  # Moderate reduction
        voxel_size_scale_range=(0.5, 2.0),  # Back to proven range
        importance_threshold=0.3,  # Back to proven threshold
    ),
    
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

# 🔥 OPTIMIZED LEARNING RATE (Fast but stable)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.018,  # Reduced from 0.025 → safer middle ground
        betas=(0.9, 0.99),  # Back to proven betas
        weight_decay=0.01,  # Back to proven weight decay
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # Back to proven clipping
)

# Balanced learning rate schedule - fast but stable
param_scheduler = [
    # Reasonable warmup
    dict(
        type='LinearLR',
        start_factor=0.5,  # Back to proven start
        by_epoch=False,
        begin=0,
        end=200,  # Back to proven 200 iterations
    ),
    # Moderate cosine annealing 
    dict(
        type='CosineAnnealingLR',
        T_max=6,  # Back to 6 epochs
        eta_min=0.002,  # Back to proven minimum
        begin=0,
        end=8,  # Back to 8 epochs
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Turbo training configuration - maximum efficiency
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)  # Reduced from 8
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimized batch size for speed vs memory balance
train_dataloader = dict(
    batch_size=2,  # Keep at 2 for memory safety
    num_workers=3,  # Increased workers for faster data loading
    persistent_workers=True,  # Keep workers alive between epochs
    pin_memory=True,  # Faster GPU transfer
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    persistent_workers=True,
    pin_memory=True
)

# Ultra-frequent monitoring for maximum insight
default_hooks = dict(
    logger=dict(interval=20),  # Log every 20 iterations (was 25)
    checkpoint=dict(interval=1, save_best='auto', max_keep_ckpts=3),  # Keep fewer checkpoints
)

work_dir = './work_dirs/adaptive_voxel_ultrafast'

# 🧠 CONSERVATIVE MEMORY OPTIMIZATION
# Set environment variable to prevent memory fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# 🎯 BALANCED TARGET PERFORMANCE (Stability + Speed):
# Current: Reset needed due to over-optimization  
# Target:  <1.5 loss by 1000 iterations (recovery)
# Expected: <1.0 loss by epoch 2
# Goal:     <0.6 loss by epoch 6

# 📊 CONSERVATIVE MONITORING EXPECTATIONS:
# Iteration 500: loss < 2.0 (recovery)
# Iteration 1000: loss < 1.5  
# Epoch 2: loss < 1.0
# Epoch 4: loss < 0.7
# Epoch 6: loss < 0.6

# 📊 MONITORING EXPECTATIONS:
# Iteration 2500: loss < 1.2
# Iteration 3000: loss < 1.0  
# Epoch 2: loss < 0.9
# Epoch 4: loss < 0.7
# Epoch 6: loss < 0.6
