"""
Generate a detailed ScaleNet architecture diagram for the research paper.
This version uses simple notation compatible with matplotlib's mathtext parser.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_scalenet_architecture():
    """Create comprehensive ScaleNet architecture visualization"""
    
    # Create figure with high resolution
    fig = plt.figure(figsize=(24, 16), dpi=300)
    ax = plt.subplot(111)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme (professional, colorblind-friendly)
    colors = {
        'input': '#E3F2FD',  # Light blue
        'learnable': '#FFF3E0',  # Light orange
        'network': '#E8F5E9',  # Light green
        'voxel': '#F3E5F5',  # Light purple
        'feature': '#FFF9C4',  # Light yellow
        'detection': '#FFEBEE',  # Light red
        'highlight': '#FF6B6B',  # Bright red
        'arrow': '#37474F'  # Dark gray
    }
    
    # Title
    fig.suptitle('ScaleNet: Learnable Multi-Scale Voxelization Architecture',
                 fontsize=20, fontweight='bold', y=0.98)
    
    y_start = 14  # Start from top
    
    # ============================================================
    # SECTION 1: INPUT POINT CLOUD (Top Left)
    # ============================================================
    
    ax.text(2, y_start + 0.8, '1. Input Point Cloud', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Input box
    input_box = FancyBboxPatch((0.5, y_start - 1.5), 3, 2, 
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    
    ax.text(2, y_start - 0.2, r'Point Cloud: $P \in \mathbb{R}^{N \times 4}$',
            fontsize=11, ha='center', fontweight='bold')
    ax.text(2, y_start - 0.6, r'$N$ points: $(x, y, z, r)$',
            fontsize=10, ha='center')
    ax.text(2, y_start - 1.0, 'r = reflectance intensity',
            fontsize=9, ha='center', style='italic')
    
    # ============================================================
    # SECTION 2: LEARNABLE PARAMETERS (Top Middle-Left)
    # ============================================================
    
    ax.text(7, y_start + 0.8, '2. Learnable Voxel Scales (PhD Contribution)', 
            fontsize=14, fontweight='bold', ha='center', color=colors['highlight'])
    
    # Parameters box
    param_box = FancyBboxPatch((5, y_start - 1.5), 4, 2, 
                               boxstyle="round,pad=0.1",
                               facecolor=colors['learnable'],
                               edgecolor=colors['highlight'], linewidth=3)
    ax.add_patch(param_box)
    
    ax.text(7, y_start - 0.2, r'Scales: $\theta = [\theta_1, \theta_2, \theta_3]$',
            fontsize=11, ha='center', fontweight='bold')
    ax.text(7, y_start - 0.6, r'Initialized: $[0.05m, 0.10m, 0.20m]$',
            fontsize=9, ha='center')
    ax.text(7, y_start - 0.95, r'Learned via gradient descent',
            fontsize=9, ha='center', style='italic')
    ax.text(7, y_start - 1.3, r'nn.Parameter (requires_grad=True)',
            fontsize=8, ha='center', color='#D32F2F', family='monospace')
    
    # Arrow from input to learnable params
    arrow1 = FancyArrowPatch((3.5, y_start - 0.5), (5, y_start - 0.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_patch(arrow1)
    ax.text(4.25, y_start - 0.2, 'Feed', fontsize=9, ha='center')
    
    # ============================================================
    # SECTION 3: SCALE ASSIGNMENT NETWORK (Top Right)
    # ============================================================
    
    ax.text(14, y_start + 0.8, 'Scale Assignment Network (ScaleNet)', 
            fontsize=14, fontweight='bold', ha='center')
    
    # ScaleNet architecture box
    scalenet_box = FancyBboxPatch((11, y_start - 1.5), 6, 2,
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['network'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(scalenet_box)
    
    # Layer details
    ax.text(14, y_start - 0.2, r'MLP: $(x,y,z,r) \to [64 \to 32 \to 3]$',
            fontsize=10, ha='center', family='monospace')
    ax.text(14, y_start - 0.6, r'Gumbel-Softmax: $\sigma = softmax(z/\tau)$',
            fontsize=10, ha='center')
    ax.text(14, y_start - 1.0, r'Output: $\sigma_i \in \mathbb{R}^3$ (per point)',
            fontsize=9, ha='center')
    ax.text(14, y_start - 1.3, 'Differentiable scale selection',
            fontsize=8, ha='center', style='italic')
    
    # Arrow from params to ScaleNet
    arrow2 = FancyArrowPatch((9, y_start - 0.5), (11, y_start - 0.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_patch(arrow2)
    ax.text(10, y_start - 0.2, 'Use', fontsize=9, ha='center')
    
    # ============================================================
    # SECTION 4: MULTI-SCALE VOXELIZATION (Middle Row)
    # ============================================================
    
    y_voxel = 10
    
    ax.text(12, y_voxel + 1.3, '4. Multi-Scale Voxelization (3 Parallel Branches)',
            fontsize=14, fontweight='bold', ha='center')
    
    # Draw three voxelization branches
    scale_names = ['Fine Scale', 'Medium Scale', 'Coarse Scale']
    scale_values = [r'$\theta_1$ (~0.04m)', r'$\theta_2$ (~0.12m)', r'$\theta_3$ (~0.25m)']
    scale_colors = ['#FFCDD2', '#FFECB3', '#C5E1A5']
    x_positions = [5, 12, 19]
    
    for i, (x_pos, name, value, color) in enumerate(zip(x_positions, scale_names, scale_values, scale_colors)):
        # Voxel box
        voxel_box = FancyBboxPatch((x_pos - 2.5, y_voxel - 0.8), 5, 1.8,
                                   boxstyle="round,pad=0.08",
                                   facecolor=color,
                                   edgecolor='black', linewidth=2)
        ax.add_patch(voxel_box)
        
        ax.text(x_pos, y_voxel + 0.7, name, fontsize=11, ha='center', fontweight='bold')
        ax.text(x_pos, y_voxel + 0.3, value, fontsize=10, ha='center')
        ax.text(x_pos, y_voxel - 0.1, f'Voxel Grid {i+1}', fontsize=9, ha='center')
        ax.text(x_pos, y_voxel - 0.5, f'Resolution: Dynamic', fontsize=8, ha='center', style='italic')
        
        # Arrow from ScaleNet to each voxel branch
        arrow_vox = FancyArrowPatch((14, y_start - 1.5), (x_pos, y_voxel + 1.0),
                                   arrowstyle='->', mutation_scale=15, linewidth=1.5,
                                   color=colors['arrow'], alpha=0.6)
        ax.add_patch(arrow_vox)
    
    # ============================================================
    # SECTION 5: SCALE-SPECIFIC VFE (Middle-Lower Row)
    # ============================================================
    
    y_vfe = 7
    
    ax.text(12, y_vfe + 1.8, '5. Scale-Specific Voxel Feature Encoding (VFE)',
            fontsize=14, fontweight='bold', ha='center')
    
    for i, x_pos in enumerate(x_positions):
        # VFE box
        vfe_box = FancyBboxPatch((x_pos - 2.5, y_vfe - 0.5), 5, 1.5,
                                 boxstyle="round,pad=0.08",
                                 facecolor=colors['feature'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(vfe_box)
        
        ax.text(x_pos, y_vfe + 0.65, f'VFE Branch {i+1}', fontsize=11, ha='center', fontweight='bold')
        ax.text(x_pos, y_vfe + 0.2, r'PointNet-style: $f^{(s)}$', fontsize=9, ha='center')
        ax.text(x_pos, y_vfe - 0.2, f'Output: Feature Map {i+1}', fontsize=8, ha='center')
        
        # Arrow from voxel to VFE
        arrow_vfe = FancyArrowPatch((x_pos, y_voxel - 0.8), (x_pos, y_vfe + 1.0),
                                   arrowstyle='->', mutation_scale=15, linewidth=2,
                                   color='black')
        ax.add_patch(arrow_vfe)
    
    # ============================================================
    # SECTION 6: WEIGHTED FEATURE AGGREGATION (Lower Middle)
    # ============================================================
    
    y_agg = 4.5
    
    ax.text(12, y_agg + 1.2, '6. Weighted Feature Aggregation',
            fontsize=14, fontweight='bold', ha='center')
    
    # Aggregation box
    agg_box = FancyBboxPatch((8, y_agg - 0.8), 8, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#E1F5FE',
                             edgecolor='black', linewidth=2)
    ax.add_patch(agg_box)
    
    ax.text(12, y_agg + 0.5, r'Weighted Sum: $f_i = \sum_{s=1}^{3} \sigma_i^{(s)} \cdot f_i^{(s)}$',
            fontsize=11, ha='center', fontweight='bold')
    ax.text(12, y_agg + 0.15, r'Differentiable w.r.t. both $\sigma$ and $\theta$',
            fontsize=9, ha='center', style='italic')
    ax.text(12, y_agg - 0.25, 'Adaptive per-point scale selection',
            fontsize=9, ha='center')
    ax.text(12, y_agg - 0.6, 'End-to-end trainable', fontsize=8, ha='center', color='#D32F2F')
    
    # Arrows from VFE to aggregation
    for x_pos in x_positions:
        arrow_agg = FancyArrowPatch((x_pos, y_vfe - 0.5), (12, y_agg + 1.0),
                                   arrowstyle='->', mutation_scale=15, linewidth=1.5,
                                   color=colors['arrow'], alpha=0.7)
        ax.add_patch(arrow_agg)
    
    # ============================================================
    # SECTION 7: DETECTION HEAD (Bottom)
    # ============================================================
    
    y_head = 1.5
    
    ax.text(12, y_head + 1.2, '7. 3D Object Detection Head',
            fontsize=14, fontweight='bold', ha='center')
    
    # Detection head box
    head_box = FancyBboxPatch((8, y_head - 0.7), 8, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['detection'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(head_box)
    
    ax.text(12, y_head + 0.5, 'Bounding Box Regression + Classification',
            fontsize=11, ha='center', fontweight='bold')
    ax.text(12, y_head + 0.15, r'Output: Boxes + Scores',
            fontsize=10, ha='center')
    ax.text(12, y_head - 0.25, r'Loss: Classification + Regression',
            fontsize=9, ha='center', style='italic')
    
    # Arrow from aggregation to detection
    arrow_det = FancyArrowPatch((12, y_agg - 0.8), (12, y_head + 0.8),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color='black')
    ax.add_patch(arrow_det)
    
    # ============================================================
    # SECTION 8: GRADIENT FLOW (Right Side)
    # ============================================================
    
    ax.text(21, 14, 'Gradient Flow', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Gradient path annotations
    gradients = [
        (20.5, 2.5, r'$\partial L / \partial f_i$'),
        (20.5, 4.5, r'$\partial L / \partial f_i^{(s)}$'),
        (20.5, 7, r'$\partial L / \partial \sigma_i$'),
        (20.5, 10, r'$\partial L / \partial z_i$'),
        (20.5, 12, r'$\partial L / \partial \theta$'),
    ]
    
    for x, y, label in gradients:
        ax.text(x, y, label, fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        # Gradient arrow
        if y < 12:
            arrow_grad = FancyArrowPatch((20.3, y + 0.3), (20.3, y + 1.5),
                                        arrowstyle='<-', mutation_scale=12, linewidth=1.5,
                                        color='red', linestyle='dashed', alpha=0.7)
            ax.add_patch(arrow_grad)
    
    ax.text(20.5, 13, 'Backpropagation', fontsize=10, ha='left', style='italic', color='red')
    
    # ============================================================
    # SECTION 9: KEY INNOVATIONS (Left Side)
    # ============================================================
    
    innovations_text = """
KEY INNOVATIONS:
    
★ Learnable voxel scales
   (PhD Contribution)
    
★ Gumbel-Softmax for
   differentiable selection
    
★ Per-point adaptive
   scale assignment
    
★ End-to-end training
   with detection loss
    
★ No manual tuning
   required
"""
    
    innovation_box = FancyBboxPatch((0.2, 4), 3.5, 5.5,
                                    boxstyle="round,pad=0.15",
                                    facecolor='#FFF9C4',
                                    edgecolor=colors['highlight'], linewidth=2)
    ax.add_patch(innovation_box)
    
    ax.text(2, 8.8, innovations_text, fontsize=9, ha='center', va='top',
            family='monospace', linespacing=1.5)
    
    # ============================================================
    # SECTION 10: PERFORMANCE METRICS (Bottom Left)
    # ============================================================
    
    metrics_text = """
PERFORMANCE:
    
AP (Car, Moderate):
• Baseline: 70.21%
• VoxAdapt: 85.09%
• Improvement: +21.2%
    
Learned Scales:
• Fine: 0.041m
• Medium: 0.118m
• Coarse: 0.251m
"""
    
    metrics_box = FancyBboxPatch((0.2, 0.3), 3.5, 3.3,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#E8F5E9',
                                 edgecolor='green', linewidth=2)
    ax.add_patch(metrics_box)
    
    ax.text(2, 3.3, metrics_text, fontsize=9, ha='center', va='top',
            family='monospace', linespacing=1.4)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Generate and save the figure
print("Creating detailed ScaleNet architecture figure...")
fig = create_scalenet_architecture()

# Save in multiple formats
output_dir = "/home/daham/mmdetection_project/mmdetection3d"
print(f"Saving figure to {output_dir}/...")

# High-resolution PNG
fig.savefig(f"{output_dir}/scalenet_architecture_detailed.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: scalenet_architecture_detailed.png (300 DPI)")

# Vector PDF (for LaTeX papers)
fig.savefig(f"{output_dir}/scalenet_architecture_detailed.pdf", 
            bbox_inches='tight', facecolor='white')
print("✓ Saved: scalenet_architecture_detailed.pdf (vector)")

# SVG (editable in Inkscape/Illustrator)
fig.savefig(f"{output_dir}/scalenet_architecture_detailed.svg", 
            bbox_inches='tight', facecolor='white')
print("✓ Saved: scalenet_architecture_detailed.svg (editable)")

print("\n✅ ScaleNet architecture figure generation complete!")
print(f"   Files saved in: {output_dir}/")
print("   Use the PDF version for your LaTeX paper.")

plt.close()
