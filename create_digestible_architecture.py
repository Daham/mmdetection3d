#!/usr/bin/env python3
"""
Create a clean, digestible VoxAdapt architecture diagram
Based on README_IN_SIMPLE_TERMS.md
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up the figure with a clean style
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Define color scheme
COLOR_INPUT = '#E8F4F8'      # Light blue
COLOR_SCALENET = '#FFE5CC'   # Light orange
COLOR_GUMBEL = '#FFE6F0'     # Light pink
COLOR_VOXEL = '#E6F3E6'      # Light green
COLOR_VFE = '#F0E6FF'        # Light purple
COLOR_FUSION = '#FFF5E6'     # Light yellow
COLOR_OUTPUT = '#FFE6E6'     # Light red

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    """Draw a rounded box with text"""
    weight = 'bold' if bold else 'normal'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         edgecolor='#333', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, label='', style='solid', color='#333'):
    """Draw an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color=color, linestyle=style)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
                fontsize=8, style='italic', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor='none', alpha=0.8))

def draw_circle_node(ax, x, y, r, text, color):
    """Draw a circular node"""
    circle = Circle((x, y), r, edgecolor='#333', facecolor=color, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')

# ============================================================================
# TITLE
# ============================================================================
ax.text(8, 9.5, 'VoxAdapt Architecture: Learnable Multi-Scale Voxelization',
        ha='center', va='center', fontsize=16, weight='bold')
ax.text(8, 9.0, 'End-to-end learnable geometric discretization for 3D object detection',
        ha='center', va='center', fontsize=11, style='italic', color='#555')

# ============================================================================
# STAGE 1: INPUT
# ============================================================================
draw_box(ax, 0.5, 7.2, 2, 1, 'Input Point Cloud\nN points\n[x, y, z, intensity]',
         COLOR_INPUT, fontsize=10, bold=True)

# ============================================================================
# STAGE 2: SCALENET (Point-wise Scale Selection)
# ============================================================================
draw_box(ax, 3.5, 6.5, 2.5, 1.5, 'ScaleNet 🧠\n(Neural Network)\n'
         'Input: Point features\nOutput: Scale logits\n[N × K]',
         COLOR_SCALENET, fontsize=9, bold=True)

# Arrow from input to ScaleNet
draw_arrow(ax, 2.5, 7.7, 3.5, 7.2, 'Extract\nfeatures')

# ============================================================================
# STAGE 3: GUMBEL-SOFTMAX (Differentiable Sampling)
# ============================================================================
draw_box(ax, 6.8, 6.5, 2.5, 1.5, 'Gumbel-Softmax 🎲\n(Differentiable)\n'
         'Logits → Probabilities\n[N × K] soft assignments',
         COLOR_GUMBEL, fontsize=9, bold=True)

# Arrow from ScaleNet to Gumbel-Softmax
draw_arrow(ax, 6.0, 7.2, 6.8, 7.2, 'logits')

# ============================================================================
# STAGE 4: LEARNABLE VOXEL SCALES (Parameters)
# ============================================================================
draw_circle_node(ax, 11.5, 7.2, 0.5, 'σ₀, σ₁, σ₂', '#FFCCCC')
ax.text(11.5, 6.4, 'Learnable\nVoxel Scales', ha='center', va='top',
        fontsize=8, weight='bold', style='italic')
ax.text(11.5, 8.0, '[0.05m, 0.10m, 0.20m]', ha='center', va='bottom',
        fontsize=7, family='monospace')

# ============================================================================
# STAGE 5: MULTI-SCALE VOXELIZATION (K parallel branches)
# ============================================================================
y_voxel_start = 3.5
scale_names = ['Fine (σ₀)', 'Medium (σ₁)', 'Coarse (σ₂)']
scale_examples = ['0.05m', '0.10m', '0.20m']
voxel_boxes = []

# Mathematical operation explanation box
math_box_y = 5.3
draw_box(ax, 0.5, math_box_y, 2.5, 1.0,
         'Voxel Index:\ni = ⌊p/σₖ⌋\n(Quantization)',
         '#FFF0E6', fontsize=8, bold=True)

for i, (name, example) in enumerate(zip(scale_names, scale_examples)):
    x_pos = 1.0 + i * 3.5
    
    # Main voxelization box with math
    box_text = (f'{name}\n{example}\n'
                f'idx = ⌊p/{example}⌋')
    box = draw_box(ax, x_pos, y_voxel_start, 2.5, 1.5,
                   box_text,
                   COLOR_VOXEL, fontsize=8, bold=True)
    voxel_boxes.append((x_pos + 1.25, y_voxel_start))
    
    # Add weighted aggregation formula below
    ax.text(x_pos + 1.25, y_voxel_start - 0.2,
            f'V[idx] += p[{i}] × f',
            ha='center', va='top', fontsize=7,
            family='monospace', color='#444',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='#888', linewidth=1))
    
    # Arrow from Gumbel-Softmax to each voxel grid
    draw_arrow(ax, 8.0, 6.5, x_pos + 1.25, y_voxel_start + 1.5,
               f'p[{i}]', color='#666')
    
    # Arrow from learnable scales to voxelization
    draw_arrow(ax, 11.5, 6.7, x_pos + 2.0, y_voxel_start + 1.2,
               '', style='dashed', color='#999')

# ============================================================================
# STAGE 6: VFE (Voxel Feature Encoding) - K parallel branches
# ============================================================================
y_vfe = 1.5
vfe_boxes = []

for i in range(3):
    x_pos = 1.0 + i * 3.5
    box_text = f'VFE_{i}\n(Sparse Conv)\nF = φ(V)'
    box = draw_box(ax, x_pos, y_vfe, 2.5, 1.2,
                   box_text,
                   COLOR_VFE, fontsize=8, bold=True)
    vfe_boxes.append((x_pos + 1.25, y_vfe))
    
    # Arrow from voxel grid to VFE
    draw_arrow(ax, voxel_boxes[i][0], voxel_boxes[i][1],
               x_pos + 1.25, y_vfe + 1.2, 'extract')

# ============================================================================
# STAGE 7: MULTI-SCALE FUSION (Attention-weighted)
# ============================================================================
draw_box(ax, 5.5, 0.3, 3, 1.0, 'Multi-Scale Fusion ⚡\n(Attention-weighted)',
         COLOR_FUSION, fontsize=10, bold=True)

# Arrows from all VFEs to fusion
for i, (x_pos, y_pos) in enumerate(vfe_boxes):
    draw_arrow(ax, x_pos, y_vfe, 7.0, 1.3, f'F{i}', color='#666')

# ============================================================================
# STAGE 8: OUTPUT
# ============================================================================
draw_box(ax, 11, 0.3, 3, 1.0, 'Detection Head 🎯\nBounding Boxes + Classes',
         COLOR_OUTPUT, fontsize=10, bold=True)

# Arrow from fusion to output
draw_arrow(ax, 8.5, 0.8, 11.0, 0.8, 'fused\nfeatures')

# ============================================================================
# GRADIENT FLOW (Backpropagation arrow)
# ============================================================================
# Gradient flow annotation - large curved arrow
ax.annotate('', xy=(1.5, 8.5), xytext=(13.0, 0.8),
            arrowprops=dict(arrowstyle='<-', lw=3, color='red',
                          linestyle='dashed', alpha=0.6))
ax.text(7.5, 4.8, '← Gradient Flow (Backpropagation)', 
        ha='center', va='center', fontsize=11, color='red',
        weight='bold', rotation=55,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='red', linewidth=2, alpha=0.9))

# ============================================================================
# KEY INSIGHTS (Annotations)
# ============================================================================
# Annotation 1: Learnable parameters
ax.text(13.5, 7.5, '✓ Trainable\nvia SGD', ha='left', va='center',
        fontsize=8, color='darkred', weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6',
                 edgecolor='darkred', linewidth=1.5))

# Annotation 2: Differentiable
ax.text(9.5, 8.0, 'Differentiable!', ha='center', va='bottom',
        fontsize=8, color='darkgreen', weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6F3E6',
                 edgecolor='darkgreen', linewidth=1.5))

# Annotation 3: Soft assignment
ax.text(5.5, 5.3, 'Soft assignments\nallow gradient flow', ha='center', va='top',
        fontsize=7, color='purple', style='italic')

# ============================================================================
# VOXELIZATION DETAIL INSET (Mathematical Operations)
# ============================================================================
inset_x, inset_y = 10.5, 3.5
inset_w, inset_h = 5.0, 3.0

# Inset border
ax.add_patch(FancyBboxPatch((inset_x, inset_y), inset_w, inset_h,
                            boxstyle="round,pad=0.15",
                            edgecolor='#0066CC', facecolor='#F0F8FF',
                            linewidth=2.5, alpha=0.95))

# Title
ax.text(inset_x + inset_w/2, inset_y + inset_h - 0.3,
        'Voxelization Mathematics',
        ha='center', va='top', fontsize=10, weight='bold', color='#0066CC')

# Step-by-step operations
steps_y = inset_y + inset_h - 0.8
step_spacing = 0.5

operations = [
    ('1. Quantization', 'idx = ⌊p/σₖ⌋', 'Map continuous → discrete'),
    ('2. Weighted Scatter', 'V[idx] += pₖ × f', 'Soft assignment (pₖ from Gumbel)'),
    ('3. Aggregation', 'V[idx] = 1/n Σ fᵢ', 'Mean pooling per voxel'),
    ('4. Feature Map', 'F = φ(V)', 'Neural network encoding'),
]

for i, (step, formula, desc) in enumerate(operations):
    y_pos = steps_y - i * step_spacing
    
    # Step label
    ax.text(inset_x + 0.2, y_pos, step,
            ha='left', va='center', fontsize=8, weight='bold', color='#333')
    
    # Formula
    ax.text(inset_x + 2.0, y_pos, formula,
            ha='left', va='center', fontsize=8, family='monospace',
            color='#0066CC', weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='#0066CC', linewidth=1))
    
    # Description
    ax.text(inset_x + 4.8, y_pos, desc,
            ha='right', va='center', fontsize=7, style='italic', color='#555')

# Key notation box at bottom of inset
notation_y = inset_y + 0.2
ax.text(inset_x + inset_w/2, notation_y,
        'p: point position | σₖ: voxel scale | pₖ: probability | f: features\n'
        'idx: 3D voxel index | V: voxel grid | φ: neural network',
        ha='center', va='bottom', fontsize=6, family='monospace',
        color='#666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6',
                 edgecolor='#999', linewidth=1))

# ============================================================================
# LEGEND
# ============================================================================
legend_y = 8.7
legend_x = 13.5

ax.text(legend_x, legend_y + 0.2, 'Legend:', ha='left', va='top',
        fontsize=10, weight='bold')

legend_items = [
    (COLOR_SCALENET, 'Learned Assignment'),
    (COLOR_GUMBEL, 'Differentiable Sampling'),
    (COLOR_VOXEL, 'Geometric Quantization'),
    (COLOR_VFE, 'Feature Extraction'),
    (COLOR_FUSION, 'Multi-Scale Fusion'),
]

for i, (color, label) in enumerate(legend_items):
    y_pos = legend_y - i * 0.35
    ax.add_patch(FancyBboxPatch((legend_x, y_pos - 0.12), 0.25, 0.22,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='#333', linewidth=1))
    ax.text(legend_x + 0.35, y_pos, label, ha='left', va='center', fontsize=7)

# ============================================================================
# KEY RESULTS BOX
# ============================================================================
results_text = (
    "Key Innovation:\n"
    "• End-to-end learnable voxel scales\n"
    "• Per-point adaptive scale selection\n"
    "• Differentiable geometric preprocessing\n\n"
    "Performance:\n"
    "• Pedestrian: 0% → 40.30% AP\n"
    "• Overhead: +0.6% params, +2.4% time"
)

ax.text(0.5, 1.0, results_text, ha='left', va='top',
        fontsize=8, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD',
                 edgecolor='#FFD700', linewidth=2))

# ============================================================================
# BOTTOM CAPTION
# ============================================================================
caption = (
    "VoxAdapt treats voxel scales (σ₀, σ₁, σ₂) as trainable parameters. "
    "ScaleNet learns to assign each point to the optimal scale, "
    "Gumbel-Softmax enables gradient flow through discrete choices, "
    "and multi-scale features are fused for final detection. "
    "All components are jointly optimized end-to-end."
)

ax.text(8, -0.3, caption, ha='center', va='top',
        fontsize=9, wrap=True, style='italic', color='#555',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                 edgecolor='#CCC', linewidth=1))

# ============================================================================
# SAVE FIGURE
# ============================================================================
plt.tight_layout()
plt.savefig('voxadapt_digestible_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('voxadapt_digestible_architecture.png', dpi=300, bbox_inches='tight')
print("✅ Digestible architecture diagram saved!")
print("   - voxadapt_digestible_architecture.pdf")
print("   - voxadapt_digestible_architecture.png")

plt.show()
