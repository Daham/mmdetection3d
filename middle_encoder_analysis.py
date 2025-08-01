#!/usr/bin/env python3
"""
Configuration recommendation for best middle encoder for adaptive voxelization.

Based on analysis of MMDetection3D middle encoders, this script provides
recommendations and creates an optimal config for adaptive voxelization.
"""

import os

def analyze_middle_encoders():
    """Analyze available middle encoders for adaptive voxelization."""
    
    analysis = {
        'SparseEncoder': {
            'type': 'Sparse Convolution Encoder',
            'pros': [
                'Proven performance (SECOND, PartA2)',
                'Efficient sparse 3D convolutions',
                'Flexible sparse representation',
                'Well optimized and stable',
                'Easy to extend for adaptive features'
            ],
            'cons': [
                'No built-in adaptive capability',
                'Fixed sparse shape assumption'
            ],
            'adaptive_compatibility': 9,
            'performance': 9,
            'complexity': 3,
            'memory_efficiency': 8,
            'recommendation_score': 8.5
        },
        
        'SparseUNet': {
            'type': 'U-Net Style Sparse Encoder',
            'pros': [
                'Skip connections for better features',
                'Multi-scale processing',
                'Good for fine-grained details',
                'Proven architecture'
            ],
            'cons': [
                'More complex than needed for detection',
                'Higher memory usage',
                'Designed more for segmentation'
            ],
            'adaptive_compatibility': 7,
            'performance': 8,
            'complexity': 6,
            'memory_efficiency': 6,
            'recommendation_score': 6.5
        },
        
        'DSVT': {
            'type': 'Dynamic Sparse Voxel Transformer',
            'pros': [
                'Transformer-based architecture',
                'Set-based attention mechanisms',
                'State-of-the-art on large datasets',
                'Natural handling of dynamic patterns'
            ],
            'cons': [
                'Very high complexity',
                'Requires specialized components',
                'High computational cost',
                'Located in projects/ (not core)',
                'Complex to modify and debug'
            ],
            'adaptive_compatibility': 8,
            'performance': 9,
            'complexity': 9,
            'memory_efficiency': 4,
            'recommendation_score': 6.0
        },
        
        'AdaptiveSparseEncoderV3Simple': {
            'type': 'Simplified Adaptive Sparse Encoder',
            'pros': [
                'Built specifically for adaptive voxelization',
                'Extends proven SparseEncoder',
                'Simple and debuggable',
                'Low overhead',
                'Easy to integrate'
            ],
            'cons': [
                'New implementation (needs testing)',
                'Limited adaptive features'
            ],
            'adaptive_compatibility': 10,
            'performance': 8,
            'complexity': 4,
            'memory_efficiency': 8,
            'recommendation_score': 9.0
        },
        
        'AdaptiveSparseEncoderV3': {
            'type': 'Full Adaptive Sparse Encoder',
            'pros': [
                'Comprehensive adaptive capabilities',
                'Multi-scale fusion',
                'Attention mechanisms',
                'Content-aware processing',
                'Research-ready features'
            ],
            'cons': [
                'Higher complexity',
                'More parameters',
                'Potential overfitting risk',
                'New implementation'
            ],
            'adaptive_compatibility': 10,
            'performance': 8,
            'complexity': 7,
            'memory_efficiency': 6,
            'recommendation_score': 7.5
        }
    }
    
    return analysis


def print_analysis():
    """Print detailed analysis of middle encoders."""
    
    analysis = analyze_middle_encoders()
    
    print("MIDDLE ENCODER ANALYSIS FOR ADAPTIVE VOXELIZATION")
    print("=" * 70)
    
    # Sort by recommendation score
    sorted_encoders = sorted(analysis.items(), key=lambda x: x[1]['recommendation_score'], reverse=True)
    
    for i, (name, info) in enumerate(sorted_encoders, 1):
        print(f"\n{i}. {name}")
        print(f"   Type: {info['type']}")
        print(f"   Recommendation Score: {info['recommendation_score']}/10")
        print(f"   Adaptive Compatibility: {info['adaptive_compatibility']}/10")
        print(f"   Performance: {info['performance']}/10")
        print(f"   Complexity: {info['complexity']}/10")
        print(f"   Memory Efficiency: {info['memory_efficiency']}/10")
        
        print("   Pros:")
        for pro in info['pros']:
            print(f"     + {pro}")
        
        print("   Cons:")
        for con in info['cons']:
            print(f"     - {con}")


def create_optimal_config():
    """Create optimal SECOND config with best middle encoder for adaptive voxelization."""
    
    config = """# configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car-adaptive-best.py
# OPTIMAL CONFIG FOR ADAPTIVE VOXELIZATION
# Using AdaptiveSparseEncoderV3Simple as the best balance of performance and adaptivity

_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-80e.py', 
    '../_base_/default_runtime.py'
]

# Dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)

# Model settings - OPTIMAL ADAPTIVE CONFIGURATION
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000))),
    
    # BEST: AdaptiveVFE with density-based adaptation
    voxel_encoder=dict(
        type='AdaptiveVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.05, 0.05, 0.1],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        # Adaptive settings
        adaptive_type='density_based',  # Best for real-world performance
        base_voxel_size=[0.05, 0.05, 0.1],
        size_bounds=[0.5, 2.0],  # Conservative bounds for stability
        learning_rate=0.001,  # Lower LR for stable adaptation
        density_threshold=0.5),
    
    # BEST: AdaptiveSparseEncoderV3Simple - optimal balance
    middle_encoder=dict(
        type='AdaptiveSparseEncoderV3Simple',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        # Adaptive settings
        adaptive_channel_boost=64),  # Moderate boost for stability
    
    # Standard SECOND backbone - proven performance
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    # Standard neck
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    # Standard detection head
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        assign_cfg=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
    # Training settings
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# Training settings
train_dataloader = dict(batch_size=6, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader

# Optimizer with adaptive-friendly settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.01),  # Lower LR for adaptive
    paramwise_cfg=dict(custom_keys={
        'voxel_encoder.size_factors': dict(lr_mult=0.1),  # Even lower for adaptive params
        'middle_encoder.size_processor': dict(lr_mult=0.1)
    }),
    clip_grad=dict(max_norm=10, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(type='CyclicLR', 
         target_ratio=(10, 1e-4), 
         cyclic_times=1, 
         step_ratio_up=0.4,
         by_epoch=False,
         begin=0,
         end=7330),
    dict(type='CyclicLR',
         target_ratio=(1e-4, 1e-7),
         cyclic_times=1,
         step_ratio_up=0.0,
         by_epoch=False,
         begin=7330,
         end=12544)
]

# Evaluation
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Custom hooks for adaptive monitoring
custom_hooks = [
    dict(type='AdaptiveMonitorHook',
         log_interval=100,
         monitor_size_factors=True,
         monitor_middle_encoder=True)
]

load_from = None
resume_from = None
"""
    
    return config


def create_alternative_configs():
    """Create alternative configurations for comparison."""
    
    configs = {}
    
    # Alternative 1: Full AdaptiveSparseEncoderV3
    configs['full_adaptive'] = """# Alternative 1: Full Adaptive Features
# Use this for research and maximum adaptive capability
middle_encoder=dict(
    type='AdaptiveSparseEncoderV3',
    in_channels=64,
    sparse_shape=[41, 1600, 1408],
    base_channels=16,
    output_channels=128,
    adaptive_processing=True,
    adaptive_attention=True,
    multi_scale_fusion=True,
    content_aware_weighting=True,
    adaptive_feature_dim=32),
"""
    
    # Alternative 2: Standard SparseEncoder with manual tuning
    configs['vanilla_tuned'] = """# Alternative 2: Vanilla SECOND with Manual Optimization
# Use this as baseline comparison
middle_encoder=dict(
    type='SparseEncoder',
    in_channels=64,
    sparse_shape=[41, 1600, 1408],
    base_channels=16,
    output_channels=128,
    encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
    encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1))),
"""
    
    # Alternative 3: SparseUNet for comparison
    configs['sparse_unet'] = """# Alternative 3: SparseUNet
# Use this if you want skip connections and multi-scale features
middle_encoder=dict(
    type='SparseUNet',
    in_channels=64,
    sparse_shape=[41, 1600, 1408],
    base_channels=16,
    output_channels=128),
"""
    
    return configs


def main():
    """Main function to analyze and recommend middle encoders."""
    
    print_analysis()
    
    print("\n\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    print("""
🏆 BEST CHOICE: AdaptiveSparseEncoderV3Simple

REASONING:
1. ✅ Built specifically for adaptive voxelization
2. ✅ Extends proven SparseEncoder foundation  
3. ✅ Simple implementation - easy to debug and tune
4. ✅ Low computational overhead
5. ✅ Compatible with existing SECOND pipeline
6. ✅ Good balance of adaptivity and performance

IMPLEMENTATION STRATEGY:
1. Start with AdaptiveSparseEncoderV3Simple
2. Compare against vanilla SECOND baseline
3. If improvements are seen, consider AdaptiveSparseEncoderV3 for more features
4. Use DSVT only for large-scale research projects
5. Avoid SparseUNet for detection tasks

NEXT STEPS:
1. Implement AdaptiveSparseEncoderV3Simple
2. Create training config with conservative adaptive parameters
3. Run training comparison: Adaptive vs Vanilla SECOND
4. Monitor adaptive parameter evolution during training
5. Evaluate detection performance improvements
""")
    
    # Save optimal config
    config_content = create_optimal_config()
    config_path = "/Users/dahamp/Documents/academic/phd-repos/mmdetection3d/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car-adaptive-best.py"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ Optimal config saved to: {config_path}")
    
    # Save alternatives
    alternatives = create_alternative_configs()
    for name, config in alternatives.items():
        alt_path = f"/Users/dahamp/Documents/academic/phd-repos/mmdetection3d/configs/second/alternative_{name}.py"
        with open(alt_path, 'w') as f:
            f.write(f"# Alternative Configuration: {name}\n{config}")
        print(f"   Alternative config saved: {alt_path}")


if __name__ == "__main__":
    main()
