#!/usr/bin/env python3
"""
Create a detailed ScaleNet architecture diagram for research paper.
This figure shows the complete pipeline of learnable voxel scale parameters.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def create_scalenet_architecture():
    """
    Create a comprehensive ScaleNet architecture figure showing:
    1. Input point cloud processing
    2. Multi-scale voxelization with learnable parameters
    3. Scale assignment network (Gumbel-Softmax)
    4. Scale-specific feature extraction
    5. Feature aggregation
    6. End-to-end gradient flow
    """
    
    # Create large figure for detailed architecture
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme (professional, colorblind-friendly)
    colors = {
        'input': '#E8F4F8',      # Light blue
        'learnable': '#FFE5E5',  # Light red (learnable params)
        'network': '#E8F5E9',    # Light green
        'voxel': '#FFF9E6',      # Light yellow
        'feature': '#F3E5F5',    # Light purple
        'output': '#E3F2FD',     # Light blue
        'gradient': '#FFEBEE',   # Light pink (gradient flow)
    }
    
    # ============================================================
    # SECTION 1: INPUT POINT CLOUD (Top Left)
    # ============================================================
    y_start = 14
    
    # Title
    ax.text(2, y_start + 0.8, 'Input: Raw Point Cloud', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Input point cloud representation
    input_box = FancyBboxPatch((0.5, y_start - 1.5), 3, 2,
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    
    ax.text(2, y_start - 0.3, r'$\mathbf{P} \in \mathbb{R}^{N \times 4}$',
            fontsize=11, ha='center', style='italic')
    ax.text(2, y_start - 0.7, r'$(x, y, z, r)$',
            fontsize=10, ha='center')
    ax.text(2, y_start - 1.1, r'$N$ points', fontsize=9, ha='center', color='gray')
    
    # ============================================================
    # SECTION 2: LEARNABLE VOXEL SCALE PARAMETERS (Top Center)
    # ============================================================
    
    # Title with PhD contribution highlight
    ax.text(7, y_start + 0.8, '⭐ Learnable Voxel Scales (PhD Contribution)', 
            fontsize=14, fontweight='bold', ha='center', color='#D32F2F')
    
    # Learnable parameter box
    param_box = FancyBboxPatch((5, y_start - 1.5), 4, 2,
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['learnable'],
                               edgecolor='#D32F2F', linewidth=3, linestyle='--')
    ax.add_patch(param_box)
    
    ax.text(7, y_start - 0.2, r'$theta_{\text{scale}} = [\theta_1, \theta_2, \theta_3]$',
            fontsize=11, ha='center', fontweight='bold')
    ax.text(7, y_start - 0.6, r'Initialized: $[0.05m, 0.10m, 0.20m]$',
            fontsize=9, ha='center')
    ax.text(7, y_start - 0.95, r'Learned via $grad_theta L_{\text{det}}$',
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
    ax.text(14, y_start - 0.2, r'MLP: $(x,y,z,r) \rightarrow [64 \rightarrow 32 \rightarrow 3]$',
            fontsize=10, ha='center', family='monospace')
    ax.text(14, y_start - 0.6, r'Output: Logits $z_i \in \mathbb{R}^3$',
            fontsize=10, ha='center')
    ax.text(14, y_start - 1.0, r'Gumbel-Softmax: $\sigma_i = \text{softmax}(\frac{z_i + \mathbf{g}}{\tau})$',
            fontsize=9, ha='center', style='italic')
    ax.text(14, y_start - 1.35, r'Temperature $\tau$: 1.0 (train) → 0.1 (test)',
            fontsize=8, ha='center', color='gray')
    
    # Arrow from learnable params to ScaleNet
    arrow2 = FancyArrowPatch((9, y_start - 0.5), (11, y_start - 0.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_patch(arrow2)
    ax.text(10, y_start - 0.2, 'Guide', fontsize=9, ha='center')
    
    # ============================================================
    # SECTION 4: MULTI-SCALE VOXELIZATION (Middle)
    # ============================================================
    
    y_mid = 9.5
    
    ax.text(12, y_mid + 2.2, 'Multi-Scale Voxelization with Learned Scales', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Arrow down from ScaleNet
    arrow3 = FancyArrowPatch((14, y_start - 1.5), (14, y_mid + 1.8),
                            arrowstyle='->', mutation_scale=25, linewidth=3,
                            color='#1976D2')
    ax.add_patch(arrow3)
    ax.text(14.8, y_mid + 3.5, r'Scale Assignment $\sigma_i$', 
            fontsize=10, ha='left', color='#1976D2', fontweight='bold')
    
    # Three parallel voxelization branches
    scales = ['Fine Scale', 'Medium Scale', 'Coarse Scale']
    scale_values = [r'$\theta_1$ (≈0.04m)', r'$\theta_2$ (≈0.12m)', r'$\theta_3$ (≈0.25m)']
    x_positions = [3, 9, 15]
    voxel_sizes = ['32³', '16³', '8³']
    
    for i, (scale, value, x_pos, vsize) in enumerate(zip(scales, scale_values, x_positions, voxel_sizes)):
        # Scale-specific box
        voxel_box = FancyBboxPatch((x_pos - 2, y_mid), 4, 1.5,
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['voxel'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(voxel_box)
        
        # Scale label
        ax.text(x_pos, y_mid + 1.2, f'{scale}', 
                fontsize=11, ha='center', fontweight='bold')
        ax.text(x_pos, y_mid + 0.85, value, 
                fontsize=10, ha='center', color='#D32F2F', fontweight='bold')
        ax.text(x_pos, y_mid + 0.5, f'Grid: {vsize}', 
                fontsize=9, ha='center', color='gray')
        ax.text(x_pos, y_mid + 0.15, f'Weight: $\sigma_i^{{({i+1})}}$', 
                fontsize=9, ha='center', style='italic')
        
        # Voxel grid visualization (small grid representation)
        grid_size = 0.15
        grid_start_x = x_pos - 0.3
        grid_start_y = y_mid + 0.02
        num_cells = [5, 4, 3][i]  # Different densities
        
        for gx in range(num_cells):
            for gy in range(num_cells):
                cell = Rectangle((grid_start_x + gx*grid_size/num_cells, 
                                grid_start_y - gy*grid_size/num_cells),
                               grid_size/num_cells, grid_size/num_cells,
                               facecolor='lightgray', edgecolor='black', linewidth=0.3)
                ax.add_patch(cell)
    
    # ============================================================
    # SECTION 5: SCALE-SPECIFIC FEATURE EXTRACTION (Lower Middle)
    # ============================================================
    
    y_lower = 6
    
    ax.text(12, y_lower + 1.8, 'Scale-Specific Voxel Feature Encoding (VFE)', 
            fontsize=14, fontweight='bold', ha='center')
    
    for i, x_pos in enumerate(x_positions):
        # Arrow down to VFE
        arrow_vfe = FancyArrowPatch((x_pos, y_mid), (x_pos, y_lower + 1.5),
                                   arrowstyle='->', mutation_scale=20, linewidth=2,
                                   color='black')
        ax.add_patch(arrow_vfe)
        
        # VFE block
        vfe_box = FancyBboxPatch((x_pos - 2, y_lower), 4, 1.3,
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['feature'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(vfe_box)
        
        ax.text(x_pos, y_lower + 0.95, f'VFE Branch {i+1}', 
                fontsize=10, ha='center', fontweight='bold')
        ax.text(x_pos, y_lower + 0.6, r'PointNet-style: $\text{MaxPool}(\phi(\mathbf{p}))$', 
                fontsize=9, ha='center', family='monospace')
        ax.text(x_pos, y_lower + 0.25, r'Output: $f_i^{(s)} \in \mathbb{R}^{C}$', 
                fontsize=9, ha='center', style='italic')
    
    # ============================================================
    # SECTION 6: WEIGHTED FEATURE AGGREGATION (Lower)
    # ============================================================
    
    y_agg = 3
    
    ax.text(12, y_agg + 1.8, 'Weighted Feature Aggregation', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Aggregation box
    agg_box = FancyBboxPatch((8, y_agg), 8, 1.5,
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2.5)
    ax.add_patch(agg_box)
    
    # Arrows from VFE to aggregation
    for x_pos in x_positions:
        arrow_agg = FancyArrowPatch((x_pos, y_lower), (12, y_agg + 1.5),
                                   arrowstyle='->', mutation_scale=20, linewidth=2,
                                   color='#1976D2', alpha=0.7)
        ax.add_patch(arrow_agg)
    
    # Aggregation formula
    ax.text(12, y_agg + 1.0, r'$f_i = \sum_{s=1}^{3} \sigma_i^{(s)} \cdot f_i^{(s)}$',
            fontsize=12, ha='center', fontweight='bold')
    ax.text(12, y_agg + 0.5, 'Soft weighted combination via Gumbel-Softmax weights',
            fontsize=10, ha='center', style='italic', color='gray')
    ax.text(12, y_agg + 0.15, r'Differentiable w.r.t. both $\sigma$ and $\theta_{\text{scale}}$',
            fontsize=9, ha='center', color='#D32F2F')
    
    # ============================================================
    # SECTION 7: DOWNSTREAM DETECTION HEAD (Bottom)
    # ============================================================
    
    y_head = 0.8
    
    # Detection head box
    head_box = FancyBboxPatch((8, y_head), 8, 1.5,
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['network'],
                             edgecolor='black', linewidth=2)
    ax.add_patch(head_box)
    
    ax.text(12, y_head + 1.15, '3D Object Detection Head', 
            fontsize=12, ha='center', fontweight='bold')
    ax.text(12, y_head + 0.75, r'Classification + Regression (BBox, Heading, Size)',
            fontsize=10, ha='center')
    ax.text(12, y_head + 0.35, r'Loss: $L_{\text{det}} = L_{\text{cls}} + L_{\text{reg}}$',
            fontsize=10, ha='center', style='italic')
    
    # Arrow from aggregation to detection head
    arrow_head = FancyArrowPatch((12, y_agg), (12, y_head + 1.5),
                                arrowstyle='->', mutation_scale=25, linewidth=3,
                                color='black')
    ax.add_patch(arrow_head)
    
    # ============================================================
    # SECTION 8: GRADIENT FLOW (Right Side - Backpropagation)
    # ============================================================
    
    ax.text(21, 8, 'Gradient Flow', fontsize=14, fontweight='bold', 
            ha='center', color='#C62828', rotation=90)
    
    # Large gradient flow arrow (bottom to top)
    grad_arrow = FancyArrowPatch((20.5, 1.5), (20.5, 13),
                                arrowstyle='<-', mutation_scale=30, linewidth=4,
                                color='#C62828', alpha=0.6, linestyle='--')
    ax.add_patch(grad_arrow)
    
    # Gradient flow annotations
    grad_labels = [
        (20.5, 2.5, r'$\frac{\partial L}{\partial f_i}$'),
        (20.5, 4.5, r'$\frac{\partial L}{\partial f_i^{(s)}}$'),
        (20.5, 7, r'$\frac{\partial L}{\partial \sigma_i}$'),
        (20.5, 10, r'$\frac{\partial L}{\partial z_i}$'),
        (20.5, 12, r'$\frac{\partial L}{\partial \theta_{\text{scale}}}$'),
    ]
    
    for x, y, label in grad_labels:
        ax.text(x + 1.5, y, label, fontsize=10, ha='left', 
               color='#C62828', style='italic', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#C62828', linewidth=1.5))
    
    # ============================================================
    # SECTION 9: KEY INNOVATIONS BOX (Left Side)
    # ============================================================
    
    innovation_box = FancyBboxPatch((0.3, 0.5), 6.5, 5,
                                   boxstyle="round,pad=0.15", 
                                   facecolor='#FFF3E0',
                                   edgecolor='#F57C00', linewidth=2.5)
    ax.add_patch(innovation_box)
    
    ax.text(3.55, 5, '🔬 Key Innovations', fontsize=12, ha='center', 
           fontweight='bold', color='#E65100')
    
    innovations = [
        '1. Learnable Scale Values (not selection)',
        '   • nn.Parameter with grad',
        '   • End-to-end optimization',
        '',
        '2. Two-Level Learning',
        '   • Policy: Which scale (Gumbel-Softmax)',
        '   • Values: What scale means (θ)',
        '',
        '3. Differentiable Voxelization',
        '   • Soft assignment via σ',
        '   • Gradients flow to geometry',
        '',
        '4. Adaptive to Dataset',
        '   • KITTI: [0.05, 0.10, 0.20]m',
        '   • Learned: [~0.04, ~0.12, ~0.25]m',
    ]
    
    y_innov = 4.5
    for i, text in enumerate(innovations):
        if text.startswith('   •'):
            ax.text(0.8, y_innov - i*0.28, text, fontsize=8, ha='left', 
                   family='monospace', color='#424242')
        elif text.startswith((' ', '\t')):
            ax.text(0.8, y_innov - i*0.28, text, fontsize=8, ha='left', color='gray')
        elif text:
            ax.text(0.8, y_innov - i*0.28, text, fontsize=9, ha='left', fontweight='bold')
    
    # ============================================================
    # SECTION 10: PERFORMANCE METRICS (Bottom Right)
    # ============================================================
    
    metrics_box = FancyBboxPatch((17.5, 0.5), 6, 3,
                                boxstyle="round,pad=0.15", 
                                facecolor='#E8F5E9',
                                edgecolor='#388E3C', linewidth=2)
    ax.add_patch(metrics_box)
    
    ax.text(20.5, 3.15, '📊 Performance Gains', fontsize=12, ha='center', 
           fontweight='bold', color='#1B5E20')
    
    metrics = [
        'KITTI Car Detection (5 epochs)',
        '─' * 30,
        'Easy:     73.76% → 84.96%  (+11.20%)',
        'Moderate: 52.34% → 73.54%  (+21.20%)',
        'Hard:     46.77% → 68.52%  (+21.75%)',
        '',
        'Overhead: Only +3.3% train time',
        'Efficiency Ratio: 5.5×',
    ]
    
    y_metric = 2.7
    for i, text in enumerate(metrics):
        if '→' in text:
            # Highlight improvements
            ax.text(17.8, y_metric - i*0.32, text, fontsize=8.5, ha='left', 
                   family='monospace', color='#1B5E20', fontweight='bold')
        elif '─' in text:
            ax.text(17.8, y_metric - i*0.32, text, fontsize=8, ha='left', color='gray')
        elif 'KITTI' in text:
            ax.text(17.8, y_metric - i*0.32, text, fontsize=9, ha='left', 
                   fontweight='bold', style='italic')
        else:
            ax.text(17.8, y_metric - i*0.32, text, fontsize=8.5, ha='left')
    
    # ============================================================
    # TITLE AND CAPTION
    # ============================================================
    
    fig.suptitle('ScaleNet: Learnable Multi-Scale Voxelization Architecture', 
                fontsize=18, fontweight='bold', y=0.98)
    
    caption = (
        'Figure: Complete architecture of ScaleNet showing learnable voxel scale parameters (red), '
        'scale assignment network with Gumbel-Softmax (green), multi-scale voxelization (yellow), '
        'scale-specific feature extraction (purple), weighted aggregation (blue), and end-to-end '
        'gradient flow (dashed red). The key innovation is making voxel scale VALUES learnable '
        'parameters optimized via backpropagation from detection loss.'
    )
    
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', 
            wrap=True, color='#424242')
    
    # ============================================================
    # LEGEND
    # ============================================================
    
    legend_elements = [
        mpatches.Patch(facecolor=colors['learnable'], edgecolor='#D32F2F', 
                      linewidth=2, label='Learnable Parameters'),
        mpatches.Patch(facecolor=colors['network'], edgecolor='black', 
                      label='Neural Network Layers'),
        mpatches.Patch(facecolor=colors['voxel'], edgecolor='black', 
                      label='Voxelization Operations'),
        mpatches.Patch(facecolor=colors['feature'], edgecolor='black', 
                      label='Feature Extraction'),
        mpatches.FancyArrow(0, 0, 0.3, 0, width=0.1, 
                          color='#C62828', 
                          label='Gradient Flow'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
             fontsize=10, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    return fig

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print("Creating detailed ScaleNet architecture figure...")
    
    fig = create_scalenet_architecture()
    
    # Save in multiple formats for paper
    output_formats = {
        'png': {'dpi': 300, 'bbox_inches': 'tight'},
        'pdf': {'bbox_inches': 'tight'},  # Vector format for LaTeX
        'svg': {'bbox_inches': 'tight'},  # Editable in Inkscape
    }
    
    for fmt, kwargs in output_formats.items():
        filename = f'scalenet_architecture_detailed.{fmt}'
        fig.savefig(filename, **kwargs)
        print(f"✓ Saved: {filename}")
    
    print("\n✅ ScaleNet architecture figure created successfully!")
    print("\nFor your paper:")
    print("  • Use PDF for LaTeX (vector graphics, infinite zoom)")
    print("  • Use PNG for Word/PowerPoint (300 DPI, high quality)")
    print("  • Use SVG if you need to edit in Inkscape/Illustrator")
    
    plt.show()
