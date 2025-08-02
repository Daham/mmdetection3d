"""
PhD Research Configuration: True Multi-Scale Adaptive Sparse Convolution

This configuration implements the revolutionary approach where:
1. Voxel sizes are learned adaptively via neural networks
2. Each voxel size group gets its own sparse convolution pathway  
3. No remapping to regular grid - maintains full adaptivity throughout!
4. Multiple pathways process different voxel sizes in parallel
5. Results are fused using attention or learnable weights

Research Innovation:
- True adaptive voxelization end-to-end
- No compromise with regular grid remapping
- Each voxel size gets specialized processing
- Maintains sparse convolution efficiency per group

Author: PhD Research Implementation  
Date: August 2025
"""

_base_ = [
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/models/second.py',
    '../_base_/schedules/cyclic-40e.py', 
    '../_base_/default_runtime.py'
]

# Multi-Scale Adaptive Sparse Model Configuration
model = dict(
    type='AdaptiveVoxelNet',  # NEW: Custom detector for adaptive processing
    
    # Learnable Adaptive Voxel Encoder (generates voxel sizes)
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        num_features=4,
        
        # Learnable voxel size parameters
        min_voxel_size=0.05,        # PhD: Minimum learnable voxel size
        max_voxel_size=0.50,        # PhD: Maximum learnable voxel size
        initial_bias=0.2,           # PhD: Initial size bias (learnable)
        
        # Neural network for size prediction
        hidden_size=64,             # PhD: Hidden layer size for size predictor
        spatial_attention_heads=4,   # PhD: Multi-head attention for spatial refinement
        
        # Feature processing
        voxel_aware_hidden=128,     # PhD: Hidden size for voxel-aware processing
    ),
    
    # Revolutionary Multi-Scale Sparse Encoder
    middle_encoder=dict(
        type='AdaptiveSparseEncoder',  # NEW: Multi-scale approach
        
        # Base sparse convolution parameters
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=['conv', 'norm', 'act'],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        
        # Sparse convolution architecture
        encoder_channels=[16, 32, 64, 64, 64, 64],
        encoder_paddings=[1, 1, 1, 1, 1, 1],
        block_type='conv_module',
        
        # PhD Research: Multi-Scale Processing
        num_size_groups=4,          # Number of parallel processing pathways
        size_group_ranges=[         # Voxel size ranges for each pathway
            (0.05, 0.15),          # Fine detail pathway (small objects)
            (0.15, 0.25),          # Medium-fine pathway  
            (0.25, 0.35),          # Medium pathway (cars, trucks)
            (0.35, 0.50)           # Coarse pathway (large structures)
        ],
        
        # Fusion strategy for combining pathways
        fusion_type='attention',    # 'attention', 'weighted', or 'concat'
    ),
    
    # Standard backbone (processes fused multi-scale features)
    backbone=dict(
        type='SECOND',
        in_channels=128,  # Matches output_channels from adaptive encoder
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    # Standard neck and head
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    
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
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='SmoothL1Loss', 
            beta=1.0 / 9.0, 
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.2
        )
    ),
    
    # Standard training and testing configs
    train_cfg=dict(
        assigner=[
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
    
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

# PhD Research: Enhanced Learning Schedule for Adaptive Components
optimizer = dict(
    type='AdamW',
    lr=0.001,      # Slightly lower LR for stable voxel size learning
    betas=(0.95, 0.99),
    weight_decay=0.01,
    
    # Different LR for adaptive components
    paramwise_cfg=dict(
        custom_keys={
            'voxel_encoder': dict(lr_mult=1.5),      # Higher LR for voxel size learning
            'middle_encoder.size_pathways': dict(lr_mult=1.2),  # Slightly higher for pathways
            'middle_encoder.fusion': dict(lr_mult=1.3),         # Higher for fusion learning
        }
    )
)

# Enhanced logging for research analysis  
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False,
        
        # PhD Research: Custom log processor for adaptive metrics
        by_epoch=False,
        log_with_hierarchy=True,
    )
)

# Research evaluation metrics
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='DefaultFormatBundle3D', class_names=['Car']),
        dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
)

# PhD Research Notes:
print("🚗 Multi-Scale Adaptive Sparse Convolution Configuration Loaded")
print("🔬 Research Features:")
print("   ✅ Learnable voxel sizes via neural networks")
print("   ✅ 4 parallel size-specific processing pathways")  
print("   ✅ Attention-based fusion of multi-scale features")
print("   ✅ No regular grid remapping - maintains full adaptivity")
print("   ✅ Enhanced learning rates for adaptive components")
print("   ✅ Research logging for voxel size analysis")
print("📊 This enables PhD research on:")
print("   - How learned voxel sizes improve detection accuracy")
print("   - What size patterns emerge for different object types")
print("   - Multi-scale feature fusion strategies")
print("   - Computational efficiency of size-specific pathways")
