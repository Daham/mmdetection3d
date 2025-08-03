# 🔥 ENHANCED Adaptive Multi-Scale Configuration with FIXED gradient flow
# Uses the new ImportanceGuidedMultiScaleVFE with aggressive scale learning

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# Model configuration with ENHANCED adaptive voxelization
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False,
        voxel_layer=None
    ),
    
    # 🔥 ENHANCED ADAPTIVE VOXELIZATION: Fixed gradient flow and scale learning
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        
        # Enhanced Multi-scale configuration  
        voxel_scales=[0.02, 0.15, 0.6],  # 🔥 30x scale diversity for strong differentiation
        num_scales=3,
        
        # Standard VFE config
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=point_cloud_range,
        
        # 🔥 ENHANCED ScaleNet configuration for better learning
        scale_net_hidden_dims=[128, 64, 32],  # Deeper network
        gumbel_temperature=5.0,  # 🔥 HIGHER initial temperature for exploration
        
        # Enhanced VFE configuration
        vfe_channels=[64, 128],  # Increased capacity
        
        # Enhanced Fusion configuration
        fusion_channels=256,
        output_channels=128,
        
        # Normalization
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)
    ),
    
    # 🚀 CRITICAL FIX: Use proper sparse encoder that matches our output
    middle_encoder=dict(
        type='SparseEncoder',  # Use standard sparse encoder
        in_channels=129,  # 128 + 1 for scale info
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128
    ),
    
    # 🚀 OPTIMIZATION: Adjusted backbone for new channel dimensions
    backbone=dict(
        type='SECOND',
        in_channels=128,  # Match middle encoder output (128 channels)
        layer_nums=[3, 5],  # 🚀 Reduced layers for speed
        layer_strides=[2, 2],  # 🚀 Simplified
        out_channels=[64, 128],  # 🚀 Reduced channels
    ),
    
    # 🚀 OPTIMIZATION: Adjusted neck
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],  # Match backbone
        upsample_strides=[1, 2],  # 🚀 Simplified
        out_channels=[128, 128],  # Consistent channels
    ),
    
    # 🚀 OPTIMIZATION: Simplified bbox head
    bbox_head=dict(
        type='Anchor3DHead',
        in_channels=256,  # 128 + 128 from neck
        feat_channels=256,  # 🚀 Reduced from 384
        num_classes=1,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True
        ),
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
        _delete_=True,  # 🚀 Clear base config
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

# 🔥 ENHANCED OPTIMIZATION: Stable training setup for adaptive voxelization
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.001,  # 🔥 REDUCED learning rate for gradient stability
        betas=(0.9, 0.999),  # 🔥 Standard betas for better convergence
        weight_decay=0.005,  # 🔥 Reduced weight decay
        eps=1e-8  # 🔥 Standard epsilon
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)  # 🔥 AGGRESSIVE gradient clipping
)

# 🔥 ENHANCED OPTIMIZATION: Adaptive learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  # 🔥 SLOWER warmup start
        by_epoch=False,
        begin=0,
        end=200,  # 🔥 LONGER warmup for stability
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=6,  # 🔥 Longer period for better convergence
        eta_min=0.0001,  # 🔥 Higher minimum LR
        begin=0,
        end=8,  # 🔥 Extended training epochs
        convert_to_iter_based=True
    )
]

# 🔥 ENHANCED Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5,  # 🔥 More epochs for convergence
    val_interval=1
)

# Data configuration (updated for stability)
train_dataloader = dict(
    batch_size=1,  # Keep batch size 1 for stability
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset'
        )
    )
)

# Evaluation configuration
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
