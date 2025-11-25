"""Visualize adaptive octree voxelization for debugging and paper figures."""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize adaptive octree voxelization')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--output-dir', default='visualizations/', help='Output directory')
    parser.add_argument('--view', choices=['bev', '3d', 'both'], default='bev', 
                       help='Visualization view')
    return parser.parse_args()


def visualize_bev_voxelization(voxel_coords, voxel_sizes, point_cloud, output_path):
    """Visualize bird's eye view of adaptive voxelization.
    
    Args:
        voxel_coords: Tensor [N, 3] - voxel center coordinates
        voxel_sizes: Tensor [N] - voxel sizes
        point_cloud: Tensor [M, 4] - raw point cloud (x, y, z, intensity)
        output_path: str - where to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert to numpy
    voxel_coords = voxel_coords.cpu().numpy()
    voxel_sizes = voxel_sizes.cpu().numpy()
    points = point_cloud.cpu().numpy()
    
    # Plot 1: Raw point cloud
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.5, cmap='viridis')
    ax.set_title('Raw Point Cloud', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Adaptive voxels colored by size
    ax = axes[1]
    scatter = ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1], 
                        c=voxel_sizes, s=20, cmap='RdYlGn_r', 
                        alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_title('Adaptive Voxelization (colored by size)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Voxel Size (m)', rotation=270, labelpad=20)
    
    # Plot 3: Voxel size distribution
    ax = axes[2]
    ax.hist(voxel_sizes, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Voxel Size Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Voxel Size (m)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = f"""Statistics:
    Total Voxels: {len(voxel_sizes)}
    Min Size: {voxel_sizes.min():.4f}m
    Max Size: {voxel_sizes.max():.4f}m
    Mean Size: {voxel_sizes.mean():.4f}m
    Median Size: {np.median(voxel_sizes):.4f}m"""
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BEV visualization to {output_path}")
    plt.close()


def visualize_voxel_grid(voxel_coords, voxel_sizes, output_path):
    """Visualize voxel grid with rectangles showing actual voxel boundaries.
    
    Args:
        voxel_coords: Tensor [N, 3] - voxel center coordinates
        voxel_sizes: Tensor [N] - voxel sizes
        output_path: str - where to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    voxel_coords = voxel_coords.cpu().numpy()
    voxel_sizes = voxel_sizes.cpu().numpy()
    
    # Draw rectangles for each voxel
    for coord, size in zip(voxel_coords, voxel_sizes):
        x, y = coord[0], coord[1]
        # Rectangle corner is at bottom-left
        rect = Rectangle((x - size/2, y - size/2), size, size,
                        linewidth=0.5, edgecolor='blue', 
                        facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)
    
    # Color voxels by size
    scatter = ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1],
                        c=voxel_sizes, s=10, cmap='RdYlGn_r',
                        edgecolors='black', linewidths=0.5, zorder=10)
    
    ax.set_title('Adaptive Voxel Grid (TRUE variable sizes)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Voxel Size (m)', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved voxel grid visualization to {output_path}")
    plt.close()


def analyze_semantic_adaptation(voxel_coords, voxel_sizes, gt_boxes, output_path):
    """Analyze if smaller voxels concentrate on objects.
    
    Args:
        voxel_coords: Tensor [N, 3] - voxel center coordinates
        voxel_sizes: Tensor [N] - voxel sizes
        gt_boxes: Tensor [M, 7] - ground truth boxes (x, y, z, dx, dy, dz, heading)
        output_path: str - where to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    voxel_coords = voxel_coords.cpu().numpy()
    voxel_sizes = voxel_sizes.cpu().numpy()
    gt_boxes = gt_boxes.cpu().numpy() if torch.is_tensor(gt_boxes) else gt_boxes
    
    # Determine which voxels are inside objects
    inside_object = np.zeros(len(voxel_coords), dtype=bool)
    
    for box in gt_boxes:
        x, y, z, dx, dy, dz, heading = box
        # Simple 2D check (BEV)
        dist_x = np.abs(voxel_coords[:, 0] - x)
        dist_y = np.abs(voxel_coords[:, 1] - y)
        inside = (dist_x < dx/2) & (dist_y < dy/2)
        inside_object |= inside
    
    # Plot voxels colored by inside/outside objects
    colors = np.where(inside_object, 'red', 'blue')
    sizes = np.where(inside_object, 30, 10)
    
    ax.scatter(voxel_coords[~inside_object, 0], 
              voxel_coords[~inside_object, 1],
              c='blue', s=10, alpha=0.3, label='Background')
    
    ax.scatter(voxel_coords[inside_object, 0],
              voxel_coords[inside_object, 1],
              c=voxel_sizes[inside_object], s=30, cmap='RdYlGn_r',
              edgecolors='black', linewidths=0.5, label='Object voxels')
    
    # Draw ground truth boxes
    for box in gt_boxes:
        x, y, z, dx, dy, dz, heading = box
        rect = Rectangle((x - dx/2, y - dy/2), dx, dy,
                        linewidth=2, edgecolor='green', 
                        facecolor='none', linestyle='--')
        ax.add_patch(rect)
    
    ax.set_title('Semantic Adaptation: Do small voxels concentrate on objects?',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Statistics
    object_sizes = voxel_sizes[inside_object]
    background_sizes = voxel_sizes[~inside_object]
    
    stats_text = f"""Semantic Statistics:
    Object voxels: {len(object_sizes)} ({100*len(object_sizes)/len(voxel_sizes):.1f}%)
    Background voxels: {len(background_sizes)} ({100*len(background_sizes)/len(voxel_sizes):.1f}%)
    
    Object mean size: {object_sizes.mean():.4f}m
    Background mean size: {background_sizes.mean():.4f}m
    
    Ratio: {background_sizes.mean() / object_sizes.mean():.2f}×
    (background should be larger)"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved semantic adaptation analysis to {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and dataset
    cfg = Config.fromfile(args.config)
    
    # Build dataset
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    
    # Get sample
    sample = dataset[args.sample_idx]
    
    print(f"Processing sample {args.sample_idx}")
    print(f"Point cloud shape: {sample['points'].shape}")
    
    # Load model and run inference
    # Note: You need to implement model loading and inference here
    # For now, we'll create dummy data for demonstration
    
    # Dummy data (replace with actual model output)
    points = sample['points']
    num_voxels = 1000
    
    # Simulate adaptive voxelization
    voxel_coords = torch.randn(num_voxels, 3) * 20  # Random voxel centers
    voxel_sizes = torch.exp(torch.randn(num_voxels) * 0.5 - 1.5)  # Log-normal sizes
    voxel_sizes = torch.clamp(voxel_sizes, 0.01, 0.6)  # Clamp to reasonable range
    
    # Get ground truth boxes if available
    gt_boxes = sample.get('gt_bboxes_3d', None)
    if gt_boxes is not None and hasattr(gt_boxes, 'tensor'):
        gt_boxes = gt_boxes.tensor
    
    # Generate visualizations
    sample_name = f"sample_{args.sample_idx}"
    
    if args.view in ['bev', 'both']:
        visualize_bev_voxelization(
            voxel_coords, voxel_sizes, points,
            output_dir / f"{sample_name}_bev.png"
        )
        
        visualize_voxel_grid(
            voxel_coords, voxel_sizes,
            output_dir / f"{sample_name}_grid.png"
        )
    
    if gt_boxes is not None:
        analyze_semantic_adaptation(
            voxel_coords, voxel_sizes, gt_boxes,
            output_dir / f"{sample_name}_semantic.png"
        )
    
    print(f"\n✅ Visualization complete! Check {output_dir}/")


if __name__ == '__main__':
    main()
