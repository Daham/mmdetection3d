# 🚀 OPTIMIZED Adaptive Voxelization Configuration - Enhanced Learning
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🚀 OPTIMIZED ADAPTIVE VOXELIZATION MODEL
model = dict(
    # Optimized voxel preprocessing
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=5,
            voxel_size=[0.08, 0.08, 0.15],  # Slightly smaller voxels for better detail
            max_voxels=(12000, 20000)       # Increased voxel count for better representation
        )
    ),
    
    # Enhanced adaptive voxel encoder with debugging
    voxel_encoder=dict(
        _delete_=True,
        type='AdaptiveLearnableVoxelLayer',
        point_cloud_range=point_cloud_range,
        base_voxel_size=[0.2, 0.2, 4.0],    # Larger base voxels for stability
        max_num_points=15,                   # Reduced for memory efficiency
        max_voxels=(5000, 10000),           # Conservative voxel count
        voxel_size_scale_range=(0.8, 1.5),  # Smaller range for stability
        importance_threshold=0.3,           # Lower threshold for more adaptation
    ),
    
    # Enhanced middle encoder with more capacity
    middle_encoder=dict(
        _delete_=True,
        type='AdaptiveVoxelEncoder',
        in_channels=4,
        out_channels=64,  # Increased capacity for better learning
    ),
    
    # Enhanced backbone
    backbone=dict(
        in_channels=64,   # Match middle_encoder output
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[32, 64, 128],  # Standard proven architecture
    ),
    
    # Enhanced neck
    neck=dict(
        in_channels=[32, 64, 128],
        upsample_strides=[1, 2, 4],
        out_channels=[64, 64, 64],  # Standard proven architecture
    ),
    
    # Enhanced detection head
    bbox_head=dict(
        in_channels=192,   # 64+64+64 from neck
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
    
    # Enhanced training configuration
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

# 🎯 ENHANCED TRAINING CONFIGURATION

# Longer training with more frequent validation
train_cfg = dict(max_epochs=15, val_interval=1)  # 15 epochs, validate every epoch

# Conservative batch size for stability
train_dataloader = dict(
    batch_size=1,  # Back to batch size 1 for stability
    num_workers=1,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
)

# 🚀 OPTIMIZED LEARNING RATE - Simple but Effective
optim_wrapper = dict(
    optimizer=dict(
        lr=0.003,  # Higher learning rate for faster convergence
        betas=(0.9, 0.99),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Override the base scheduler with a more aggressive one
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=8,  # 8 epochs cycle
        eta_min=0.0001,  # Minimum learning rate
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Enhanced logging and monitoring
default_hooks = dict(
    logger=dict(interval=20),  # Log every 20 iterations
    checkpoint=dict(interval=2, save_best='auto'),  # Save best model automatically
    param_scheduler=dict(type='ParamSchedulerHook'),
)

# Enhanced work directory with better naming
work_dir = './work_dirs/adaptive_voxel_optimized'

# 📊 TRAINING MONITORING TIPS:
# Watch for these improvements:
# 1. loss_cls should decrease from 0.74 to < 0.5
# 2. loss_bbox should decrease from 1.4 to < 1.0  
# 3. Total loss should decrease from 2.3 to < 1.8
# 4. Validation mAP should increase over epochs

# 🎯 EXPECTED IMPROVEMENTS:
# - Faster convergence due to higher learning rate
# - Better feature representation with increased capacity
# - More adaptive voxelization with lower importance threshold
# - Better gradient flow with optimized scheduler
