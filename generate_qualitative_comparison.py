#!/usr/bin/env python3
"""
Generate Qualitative Detection Examples for Research Paper
Compares VoxAdapt (Method 3) vs Baselines

Creates side-by-side visualizations showing:
- Ground Truth
- Baseline Single-Scale Detection
- VoxAdapt (Learnable Multi-Scale) Detection
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate qualitative comparison figures for paper')
    parser.add_argument(
        '--baseline-config',
        default='configs/second/validation_baseline_01_single_scale_80ep.py',
        help='Config file for baseline model')
    parser.add_argument(
        '--baseline-checkpoint',
        default='work_dirs/comparison_5epochs/method1_single/epoch_5.pth',
        help='Checkpoint file for baseline model (optional - if not provided, will show GT vs VoxAdapt only)')
    parser.add_argument(
        '--voxadapt-config',
        default='configs/second/baseline_03_adaptive_multiscale_learnable.py',
        help='Config file for VoxAdapt model')
    parser.add_argument(
        '--voxadapt-checkpoint',
        default='work_dirs/method3_with_fix_5epochs/epoch_5.pth',
        help='Checkpoint file for VoxAdapt model')
    parser.add_argument(
        '--data-root',
        default='/home/daham/mmdetection_project/dataset/KITTI',
        help='KITTI dataset root')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=6,
        help='Number of samples to visualize')
    parser.add_argument(
        '--output-dir',
        default='qualitative_results',
        help='Output directory for figures')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Score threshold for detections')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device for inference')
    parser.add_argument(
        '--sample-indices',
        type=int,
        nargs='+',
        help='Specific sample indices to visualize (e.g., 0 10 20 30)')
    
    return parser.parse_args()


def get_kitti_val_samples(data_root, num_samples=6, sample_indices=None):
    """Get validation samples from KITTI dataset."""
    val_info_path = Path(data_root) / 'kitti_infos_val.pkl'
    
    if not val_info_path.exists():
        print(f"Warning: {val_info_path} not found. Using default samples.")
        # Return some default sample IDs
        return [f"{i:06d}" for i in range(num_samples)], None
    
    import pickle
    with open(val_info_path, 'rb') as f:
        infos = pickle.load(f)
    
    # Get sample IDs
    if sample_indices is not None:
        sample_ids = [infos['data_list'][i]['sample_idx'] for i in sample_indices]
    else:
        # Select evenly spaced samples
        step = len(infos['data_list']) // num_samples
        sample_ids = [infos['data_list'][i * step]['sample_idx'] 
                     for i in range(num_samples)]
    
    # Format sample IDs as 6-digit strings if they're integers
    sample_ids = [f"{sid:06d}" if isinstance(sid, int) else sid for sid in sample_ids]
    
    return sample_ids, infos


def run_inference_on_sample(inferencer, pcd_path, score_thr=0.3):
    """Run inference on a single point cloud sample."""
    result = inferencer(
        inputs=dict(points=pcd_path),
        pred_score_thr=score_thr,
        no_save_vis=True,
        no_save_pred=True,
        out_dir=''
    )
    return result


def project_3d_to_bev(points, boxes_3d, xlim=(-40, 40), ylim=(0, 70)):
    """Project 3D points and boxes to Bird's Eye View."""
    # Points: [N, 4] with x, y, z, intensity
    # Boxes: [M, 7] with x, y, z, dx, dy, dz, yaw
    
    bev_points = points[:, :2]  # x, y only
    
    # Filter points within BEV range
    mask = ((bev_points[:, 0] >= xlim[0]) & (bev_points[:, 0] <= xlim[1]) &
            (bev_points[:, 1] >= ylim[0]) & (bev_points[:, 1] <= ylim[1]))
    bev_points = bev_points[mask]
    
    return bev_points, boxes_3d


def draw_bev_boxes(ax, boxes_3d, scores, labels, class_names, 
                   color='red', label_prefix=''):
    """Draw 3D boxes in BEV projection."""
    if len(boxes_3d) == 0:
        return
    
    for box, score, label in zip(boxes_3d, scores, labels):
        # Box format: [x, y, z, dx, dy, dz, yaw]
        x, y, z, dx, dy, dz, yaw = box
        
        # Create rotated rectangle for BEV
        corners = np.array([
            [-dx/2, -dy/2],
            [dx/2, -dy/2],
            [dx/2, dy/2],
            [-dx/2, dy/2]
        ])
        
        # Rotate corners
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        corners = corners @ rot_matrix.T
        corners[:, 0] += x
        corners[:, 1] += y
        
        # Draw box
        poly = patches.Polygon(
            corners, fill=False, edgecolor=color, linewidth=3, alpha=0.9
        )
        ax.add_patch(poly)
        
        # Add label with larger font
        class_name = class_names[label] if label < len(class_names) else f'C{label}'
        text = f'{label_prefix}{class_name}: {score:.2f}'
        ax.text(x, y, text, fontsize=14, color=color, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))


def create_comparison_figure(sample_id, pcd_path, gt_info, 
                            baseline_result, voxadapt_result,
                            output_path, score_thr=0.3):
    """Create a 3-panel comparison figure."""
    
    # Load point cloud
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    
    # Create figure with 3 subplots - MUCH LARGER for visibility
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f'Sample {sample_id}: Detection Comparison', 
                 fontsize=24, fontweight='bold')
    
    xlim = (-40, 40)
    ylim = (0, 70)
    
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    
    # Ground Truth
    ax = axes[0]
    bev_points, _ = project_3d_to_bev(points, None, xlim, ylim)
    ax.scatter(bev_points[:, 0], bev_points[:, 1], s=1.0, c='gray', alpha=0.4)
    
    if gt_info is not None:
        # Extract ground truth from KITTI info structure
        if 'instances' in gt_info and len(gt_info['instances']) > 0:
            gt_boxes = []
            gt_labels = []
            for inst in gt_info['instances']:
                if 'bbox_3d' in inst:
                    bbox = inst['bbox_3d']
                    gt_boxes.append(bbox)
                    gt_labels.append(inst['bbox_label_3d'])
            
            if len(gt_boxes) > 0:
                gt_boxes = np.array(gt_boxes)
                gt_labels = np.array(gt_labels)
                gt_scores = np.ones(len(gt_boxes))
                draw_bev_boxes(ax, gt_boxes, gt_scores, gt_labels, class_names, 
                              color='green', label_prefix='GT ')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=16, fontweight='bold')
    ax.set_title('Ground Truth', fontsize=20, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(labelsize=14)
    
    # Baseline
    ax = axes[1]
    ax.scatter(bev_points[:, 0], bev_points[:, 1], s=1.0, c='gray', alpha=0.4)
    
    if baseline_result is not None:
        pred = baseline_result['predictions'][0]
        # Handle both dict and object formats
        if isinstance(pred, dict):
            if 'bboxes_3d' in pred:
                pred_boxes = np.array(pred['bboxes_3d'])
                pred_scores = np.array(pred['scores_3d'])
                pred_labels = np.array(pred['labels_3d'])
        elif hasattr(pred, 'pred_instances_3d'):
            pred_inst = pred.pred_instances_3d
            pred_boxes = pred_inst.bboxes_3d.tensor.cpu().numpy()
            pred_scores = pred_inst.scores_3d.cpu().numpy()
            pred_labels = pred_inst.labels_3d.cpu().numpy()
        else:
            pred_boxes = pred_scores = pred_labels = None
        
        if pred_boxes is not None and len(pred_boxes) > 0:
            # Filter by score threshold
            mask = pred_scores >= score_thr
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
            
            draw_bev_boxes(ax, pred_boxes, pred_scores, pred_labels, class_names,
                          color='blue', label_prefix='')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=16, fontweight='bold')
    ax.set_title('Baseline (Single-Scale)', fontsize=20, fontweight='bold', color='blue')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(labelsize=14)
    
    # VoxAdapt
    ax = axes[2]
    ax.scatter(bev_points[:, 0], bev_points[:, 1], s=1.0, c='gray', alpha=0.4)
    
    if voxadapt_result is not None:
        pred = voxadapt_result['predictions'][0]
        # Handle both dict and object formats
        if isinstance(pred, dict):
            if 'bboxes_3d' in pred:
                pred_boxes = np.array(pred['bboxes_3d'])
                pred_scores = np.array(pred['scores_3d'])
                pred_labels = np.array(pred['labels_3d'])
        elif hasattr(pred, 'pred_instances_3d'):
            pred_inst = pred.pred_instances_3d
            pred_boxes = pred_inst.bboxes_3d.tensor.cpu().numpy()
            pred_scores = pred_inst.scores_3d.cpu().numpy()
            pred_labels = pred_inst.labels_3d.cpu().numpy()
        else:
            pred_boxes = pred_scores = pred_labels = None
        
        if pred_boxes is not None and len(pred_boxes) > 0:
            # Filter by score threshold
            mask = pred_scores >= score_thr
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
            
            draw_bev_boxes(ax, pred_boxes, pred_scores, pred_labels, class_names,
                          color='red', label_prefix='')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=16, fontweight='bold')
    ax.set_title('VoxAdapt (Ours)', fontsize=20, fontweight='bold', color='red')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison figure: {output_path}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Qualitative Comparison Figures for Research Paper")
    print("=" * 80)
    print(f"Baseline: {args.baseline_checkpoint}")
    print(f"VoxAdapt: {args.voxadapt_checkpoint}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Initialize inferencers
    baseline_inferencer = None
    if args.baseline_checkpoint and Path(args.baseline_checkpoint).exists():
        print("\n[1/4] Initializing baseline model...")
        baseline_inferencer = LidarDet3DInferencer(
            model=args.baseline_config,
            weights=args.baseline_checkpoint,
            device=args.device
        )
    else:
        print("\n[1/4] Skipping baseline model (checkpoint not provided or not found)")
        if args.baseline_checkpoint:
            print(f"      Checkpoint not found: {args.baseline_checkpoint}")
    
    print("[2/4] Initializing VoxAdapt model...")
    voxadapt_inferencer = LidarDet3DInferencer(
        model=args.voxadapt_config,
        weights=args.voxadapt_checkpoint,
        device=args.device
    )
    
    # Get validation samples
    print(f"[3/4] Loading KITTI validation samples...")
    sample_ids, infos = get_kitti_val_samples(
        args.data_root, 
        args.num_samples,
        args.sample_indices
    )
    
    print(f"Selected samples: {sample_ids}")
    
    # Process each sample
    print(f"[4/4] Generating {len(sample_ids)} comparison figures...")
    for idx, sample_id in enumerate(sample_ids):
        print(f"\n  Processing sample {idx+1}/{len(sample_ids)}: {sample_id}")
        
        # Get point cloud path
        pcd_path = Path(args.data_root) / 'training' / 'velodyne' / f'{sample_id}.bin'
        
        if not pcd_path.exists():
            print(f"    Warning: Point cloud not found: {pcd_path}")
            continue
        
        # Get ground truth info
        gt_info = None
        if infos is not None:
            # Convert sample_id to int for comparison
            sample_id_int = int(sample_id) if isinstance(sample_id, str) else sample_id
            for info in infos['data_list']:
                if info['sample_idx'] == sample_id_int:
                    gt_info = info
                    break
        
        # Run inference
        baseline_result = None
        if baseline_inferencer:
            print(f"    Running baseline inference...")
            baseline_result = run_inference_on_sample(
                baseline_inferencer, str(pcd_path), args.score_thr
            )
        
        print(f"    Running VoxAdapt inference...")
        voxadapt_result = run_inference_on_sample(
            voxadapt_inferencer, str(pcd_path), args.score_thr
        )
        
        # Create comparison figure
        output_path = output_dir / f'comparison_{sample_id}.png'
        create_comparison_figure(
            sample_id, str(pcd_path), gt_info,
            baseline_result, voxadapt_result,
            str(output_path), args.score_thr
        )
    
    print("\n" + "=" * 80)
    print(f"✓ Generated {len(sample_ids)} comparison figures in: {output_dir}")
    print("=" * 80)
    print("\nYou can now use these figures in your research paper!")
    print("Suggested caption:")
    print("""
Figure X: Qualitative detection results on KITTI validation set. 
(Left) Ground truth annotations. (Middle) Baseline single-scale detector. 
(Right) VoxAdapt with learnable multi-scale voxelization (ours). 
VoxAdapt demonstrates improved detection of distant and small objects 
through adaptive scale selection.
    """)


if __name__ == '__main__':
    main()
