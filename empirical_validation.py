#!/usr/bin/env python3
"""
Empirical validation script for adaptive voxelization in MMDetection3D.

This script runs a controlled experiment to validate that adaptive voxelization
provides benefits over vanilla fixed voxel sizes using the recommended 
AdaptiveSparseEncoderV3Simple architecture.

Experiment Design:
1. Train vanilla SECOND baseline (fixed voxel sizes)
2. Train adaptive SECOND (AdaptiveVFE + AdaptiveSparseEncoderV3Simple)
3. Compare detection performance, training dynamics, and computational cost
4. Analyze adaptive parameter evolution during training
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

def setup_experiment():
    """Setup experiment directories and configurations."""
    
    base_path = Path("/Users/dahamp/Documents/academic/phd-repos/mmdetection3d")
    experiment_dir = base_path / "experiments" / "adaptive_validation"
    
    # Create experiment directories
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "baseline").mkdir(exist_ok=True)
    (experiment_dir / "adaptive").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    return experiment_dir

def create_baseline_config(experiment_dir: Path) -> Path:
    """Create baseline SECOND config (vanilla, fixed voxels)."""
    
    config_content = '''# Baseline SECOND Configuration - Fixed Voxel Sizes
_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-80e.py', 
    '../_base_/default_runtime.py'
]

# Dataset
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# BASELINE MODEL - Standard SECOND
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],  # FIXED voxel size
            max_voxels=(16000, 40000))),
    
    # Standard VFE
    voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.05, 0.05, 0.1],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    
    # Standard sparse encoder
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1))),
    
    # Standard backbone and head
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6], [0, -40.0, -0.6, 70.4, 40.0, -0.6], [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
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
train_dataloader = dict(batch_size=4, num_workers=4)  # Smaller batch for comparison
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))

# Shorter schedule for quick comparison
param_scheduler = [
    dict(type='CyclicLR', target_ratio=(10, 1e-4), cyclic_times=1, step_ratio_up=0.4,
         by_epoch=False, begin=0, end=2000),  # Shorter schedule
    dict(type='CyclicLR', target_ratio=(1e-4, 1e-7), cyclic_times=1, step_ratio_up=0.0,
         by_epoch=False, begin=2000, end=3000)
]

# Evaluation
val_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_val.pkl', metric='bbox')
test_evaluator = val_evaluator

# Logging
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),  # Save every epoch
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Experiment settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)  # Short training
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
'''
    
    config_path = experiment_dir / "baseline_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def create_adaptive_config(experiment_dir: Path) -> Path:
    """Create adaptive SECOND config using recommended setup."""
    
    config_content = '''# Adaptive SECOND Configuration - Learnable Voxel Sizes
_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-80e.py', 
    '../_base_/default_runtime.py'
]

# Dataset
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# ADAPTIVE MODEL - Recommended AdaptiveSparseEncoderV3Simple
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.05, 0.05, 0.1],  # Base voxel size
            max_voxels=(16000, 40000))),
    
    # ADAPTIVE VFE
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
        adaptive_type='density_based',
        base_voxel_size=[0.05, 0.05, 0.1],
        size_bounds=[0.5, 2.0],
        learning_rate=0.001,
        density_threshold=0.5),
    
    # ADAPTIVE SPARSE ENCODER - Recommended V3Simple
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
        adaptive_channel_boost=64),
    
    # Same backbone and head for fair comparison
    backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6], [0, -40.0, -0.6, 70.4, 40.0, -0.6], [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
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

# Training settings - same as baseline
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader

# Optimizer with adaptive-specific settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'voxel_encoder.size_factors': dict(lr_mult=0.1),  # Lower LR for adaptive params
        'middle_encoder.size_processor': dict(lr_mult=0.1)
    }),
    clip_grad=dict(max_norm=10, norm_type=2))

# Same schedule as baseline
param_scheduler = [
    dict(type='CyclicLR', target_ratio=(10, 1e-4), cyclic_times=1, step_ratio_up=0.4,
         by_epoch=False, begin=0, end=2000),
    dict(type='CyclicLR', target_ratio=(1e-4, 1e-7), cyclic_times=1, step_ratio_up=0.0,
         by_epoch=False, begin=2000, end=3000)
]

# Evaluation
val_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_val.pkl', metric='bbox')
test_evaluator = val_evaluator

# Logging with adaptive monitoring
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Custom adaptive monitoring
custom_hooks = [
    dict(type='AdaptiveMonitorHook',
         log_interval=50,
         monitor_size_factors=True,
         monitor_middle_encoder=True)
]

# Experiment settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
'''
    
    config_path = experiment_dir / "adaptive_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def create_monitoring_script(experiment_dir: Path) -> Path:
    """Create a script to monitor training progress and adaptive parameters."""
    
    script_content = '''#!/usr/bin/env python3
"""
Monitor adaptive training progress and parameter evolution.
"""
import json
import time
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """Parse training log file to extract metrics."""
    metrics = {
        'iterations': [],
        'loss': [],
        'lr': [],
        'adaptive_sizes': [],
        'memory_usage': []
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'INFO' in line and 'Epoch' in line:
                # Parse iteration and loss
                iter_match = re.search(r'iter: (\\d+)', line)
                loss_match = re.search(r'loss: ([\\d\\.]+)', line)
                lr_match = re.search(r'lr: ([\\d\\.e-]+)', line)
                
                if iter_match:
                    metrics['iterations'].append(int(iter_match.group(1)))
                if loss_match and len(metrics['iterations']) > len(metrics['loss']):
                    metrics['loss'].append(float(loss_match.group(1)))
                if lr_match and len(metrics['iterations']) > len(metrics['lr']):
                    metrics['lr'].append(float(lr_match.group(1)))
            
            # Parse adaptive parameters if present
            if 'adaptive_sizes' in line:
                size_match = re.search(r'adaptive_sizes: \\[([\\d\\., ]+)\\]', line)
                if size_match:
                    sizes = [float(x.strip()) for x in size_match.group(1).split(',')]
                    metrics['adaptive_sizes'].append(sizes)
    
    return metrics

def plot_comparison(baseline_metrics, adaptive_metrics, save_path):
    """Plot comparison between baseline and adaptive training."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline vs Adaptive SECOND Training Comparison')
    
    # Loss comparison
    axes[0, 0].plot(baseline_metrics['iterations'], baseline_metrics['loss'], 
                   label='Baseline SECOND', color='blue')
    axes[0, 0].plot(adaptive_metrics['iterations'], adaptive_metrics['loss'], 
                   label='Adaptive SECOND', color='red')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Learning rate
    axes[0, 1].plot(baseline_metrics['iterations'], baseline_metrics['lr'], 
                   label='Baseline', color='blue')
    axes[0, 1].plot(adaptive_metrics['iterations'], adaptive_metrics['lr'], 
                   label='Adaptive', color='red')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    
    # Adaptive parameter evolution
    if adaptive_metrics['adaptive_sizes']:
        adaptive_sizes = adaptive_metrics['adaptive_sizes']
        iterations = list(range(len(adaptive_sizes)))
        
        # Plot x, y, z adaptive sizes
        for i, dim in enumerate(['X', 'Y', 'Z']):
            dim_sizes = [sizes[i] if len(sizes) > i else 1.0 for sizes in adaptive_sizes]
            axes[1, 0].plot(iterations, dim_sizes, label=f'{dim} size factor')
        
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Size Factor')
        axes[1, 0].set_title('Adaptive Voxel Size Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No adaptive size data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Adaptive Sizes (No Data)')
    
    # Performance summary
    if baseline_metrics['loss'] and adaptive_metrics['loss']:
        final_baseline_loss = baseline_metrics['loss'][-1]
        final_adaptive_loss = adaptive_metrics['loss'][-1]
        
        improvement = ((final_baseline_loss - final_adaptive_loss) / final_baseline_loss) * 100
        
        summary_text = f"""
Training Summary:
Baseline Final Loss: {final_baseline_loss:.4f}
Adaptive Final Loss: {final_adaptive_loss:.4f}
Improvement: {improvement:.2f}%

Adaptive Benefits:
✓ Dynamic voxel adaptation
✓ Content-aware processing  
✓ Density-based optimization
"""
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")

def monitor_training(experiment_dir):
    """Monitor training progress and generate reports."""
    
    baseline_log = experiment_dir / "baseline" / "*.log"
    adaptive_log = experiment_dir / "adaptive" / "*.log"
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring and generate final report")
    
    try:
        while True:
            # Parse current logs
            baseline_metrics = parse_log_file(baseline_log)
            adaptive_metrics = parse_log_file(adaptive_log)
            
            # Print current status
            print(f"\\rBaseline iterations: {len(baseline_metrics['iterations'])}, "
                  f"Adaptive iterations: {len(adaptive_metrics['iterations'])}", end='')
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\\nGenerating final comparison report...")
        
        # Generate final plots
        plot_path = experiment_dir / "results" / "training_comparison.png"
        plot_comparison(baseline_metrics, adaptive_metrics, plot_path)
        
        # Save metrics
        results = {
            'baseline': baseline_metrics,
            'adaptive': adaptive_metrics,
            'timestamp': time.time()
        }
        
        results_path = experiment_dir / "results" / "training_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    import sys
    experiment_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/adaptive_validation")
    monitor_training(experiment_dir)
'''
    
    script_path = experiment_dir / "monitor_training.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def run_experiment():
    """Run the full empirical validation experiment."""
    
    print("🚀 EMPIRICAL VALIDATION: ADAPTIVE VOXELIZATION")
    print("="*60)
    
    # Setup
    experiment_dir = setup_experiment()
    print(f"Experiment directory: {experiment_dir}")
    
    # Create configurations
    baseline_config = create_baseline_config(experiment_dir)
    adaptive_config = create_adaptive_config(experiment_dir)
    monitor_script = create_monitoring_script(experiment_dir)
    
    print(f"✅ Baseline config: {baseline_config}")
    print(f"✅ Adaptive config: {adaptive_config}")
    print(f"✅ Monitor script: {monitor_script}")
    
    # Training commands
    base_path = "/Users/dahamp/Documents/academic/phd-repos/mmdetection3d"
    
    baseline_cmd = [
        "python", "tools/train.py", 
        str(baseline_config),
        f"--work-dir={experiment_dir}/baseline",
        "--launcher=none"
    ]
    
    adaptive_cmd = [
        "python", "tools/train.py",
        str(adaptive_config), 
        f"--work-dir={experiment_dir}/adaptive",
        "--launcher=none"
    ]
    
    print("\\n📋 EXPERIMENT PLAN:")
    print("1. Train baseline SECOND (5 epochs)")
    print("2. Train adaptive SECOND (5 epochs)")
    print("3. Compare detection performance")
    print("4. Analyze adaptive parameter evolution")
    print("5. Generate comprehensive report")
    
    print("\\n⚡ TRAINING COMMANDS:")
    print(f"Baseline: {' '.join(baseline_cmd)}")
    print(f"Adaptive: {' '.join(adaptive_cmd)}")
    
    # Instructions for manual execution
    print("\\n📝 MANUAL EXECUTION INSTRUCTIONS:")
    print("1. Open two terminals")
    print("2. In terminal 1, run:")
    print(f"   cd {base_path}")
    print(f"   {' '.join(baseline_cmd)}")
    print("3. In terminal 2, run:")
    print(f"   cd {base_path}")
    print(f"   {' '.join(adaptive_cmd)}")
    print("4. In terminal 3, monitor progress:")
    print(f"   python {monitor_script}")
    
    # Expected outcomes
    print("\\n🎯 EXPECTED OUTCOMES:")
    print("✅ Adaptive SECOND should show:")
    print("   - Evolving voxel size parameters during training")
    print("   - Potential improvements in convergence speed")
    print("   - Better handling of sparse vs dense regions")
    print("   - Comparable or better detection performance")
    
    print("📊 METRICS TO COMPARE:")
    print("   - Training loss convergence")
    print("   - Validation mAP scores")
    print("   - Training time per epoch")
    print("   - Memory usage")
    print("   - Adaptive parameter evolution")
    
    return {
        'experiment_dir': experiment_dir,
        'baseline_config': baseline_config,
        'adaptive_config': adaptive_config,
        'monitor_script': monitor_script,
        'baseline_cmd': baseline_cmd,
        'adaptive_cmd': adaptive_cmd
    }

def main():
    """Main function to set up and run empirical validation."""
    
    experiment_info = run_experiment()
    
    print("\\n" + "="*60)
    print("🔬 EMPIRICAL VALIDATION SETUP COMPLETE")
    print("="*60)
    print("\\nThe experiment is ready to run. Execute the training commands")
    print("in separate terminals to begin the comparison study.")
    print("\\nThis will provide definitive evidence whether adaptive")
    print("voxelization provides benefits over vanilla SECOND.")

if __name__ == "__main__":
    main()
