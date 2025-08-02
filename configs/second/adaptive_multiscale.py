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
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/schedules/cyclic-40e.py', 
    '../_base_/default_runtime.py'
]

# Multi-Scale Adaptive Sparse Model Configuration
model = dict(
    type='AdaptiveVoxelNet',  # NEW: Custom detector for adaptive processing
    
    # Override the voxel encoder with our adaptive one
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        num_features=4,
        
        # Learnable voxel size parameters
        min_voxel_size=0.05,        # PhD: Minimum learnable voxel size
        max_voxel_size=0.50,        # PhD: Maximum learnable voxel size
        initial_bias=0.2,           # PhD: Initial size bias (learnable)
        
        # Neural network for size prediction
        voxel_predictor_hidden=128,  # PhD: Hidden layer size for size predictor
        spatial_encoding_dim=64,     # PhD: Spatial feature encoding dimension
        
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
    
    # Override backbone to match our output channels
    backbone=dict(
        type='SECOND',
        in_channels=128,  # Matches output_channels from adaptive encoder
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    # Override bbox_head for single class (car)
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,  # Only car detection
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6]],
            sizes=[[3.9, 1.6, 1.56]],  # Car size
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
            beta=1.0 / 9.0, 
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.2
        )
    ),
)

# PhD Research: Enhanced Learning Schedule for Adaptive Components
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.001,      # Slightly lower LR for stable voxel size learning
        betas=(0.95, 0.99),
        weight_decay=0.01
    ),
    # Different LR for adaptive components
    paramwise_cfg=dict(
        custom_keys={
            'voxel_encoder': dict(lr_mult=1.5),      # Higher LR for voxel size learning
            'middle_encoder.size_pathways': dict(lr_mult=1.2),  # Slightly higher for pathways
            'middle_encoder.fusion': dict(lr_mult=1.3),         # Higher for fusion learning
        }
    )
)

# Enhanced training/testing configs - inherit from base schedule
# No need to override train_cfg, val_cfg, test_cfg - they come from cyclic-40e.py

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
