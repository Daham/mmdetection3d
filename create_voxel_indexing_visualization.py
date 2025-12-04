#!/usr/bin/env python3
"""
Create visual illustration of voxel indexing and generalization
Shows how floor division maps continuous points to discrete voxels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# SUBPLOT 1: 2D Voxel Indexing Illustration (Top)
# ============================================================================
ax1 = plt.subplot(2, 3, (1, 2))
ax1.set_xlim(5.0, 5.5)
ax1.set_ylim(3.5, 4.0)
ax1.set_xlabel('X coordinate (meters)', fontsize=11, weight='bold')
ax1.set_ylabel('Y coordinate (meters)', fontsize=11, weight='bold')
ax1.set_title('Voxel Indexing: Continuous Points → Discrete Grid\n(2D View, Z slice)', 
              fontsize=13, weight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# Draw voxel grid (0.10m cells)
voxel_size = 0.10
colors_grid = ['#E8F4F8', '#FFF5E6', '#E6F3E6', '#FFE6F0']

for i, x in enumerate(np.arange(5.0, 5.5, voxel_size)):
    for j, y in enumerate(np.arange(3.5, 4.0, voxel_size)):
        color = colors_grid[(i + j) % len(colors_grid)]
        rect = Rectangle((x, y), voxel_size, voxel_size, 
                         facecolor=color, edgecolor='#333', linewidth=1.5)
        ax1.add_patch(rect)
        
        # Add voxel index labels
        voxel_idx_x = int(np.floor(x / voxel_size))
        voxel_idx_y = int(np.floor(y / voxel_size))
        ax1.text(x + 0.05, y + 0.05, f'({voxel_idx_x},{voxel_idx_y})',
                ha='center', va='center', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='none', alpha=0.7))

# Add example points that map to same voxel
target_voxel_x, target_voxel_y = 52, 36
example_points = [
    (5.23, 3.67, 0.8, 'P₁'),
    (5.21, 3.65, 0.7, 'P₂'),
    (5.28, 3.69, 0.9, 'P₃'),
    (5.25, 3.62, 0.75, 'P₄'),
    (5.29, 3.68, 0.85, 'P₅'),
]

for x, y, intensity, label in example_points:
    # Plot point
    ax1.plot(x, y, 'ro', markersize=10, zorder=5)
    ax1.text(x, y + 0.02, label, ha='center', va='bottom', 
            fontsize=9, weight='bold', color='darkred')
    
    # Show calculation
    idx_x = int(np.floor(x / voxel_size))
    idx_y = int(np.floor(y / voxel_size))

# Highlight the target voxel
target_x = target_voxel_x * voxel_size
target_y = target_voxel_y * voxel_size
highlight = Rectangle((target_x, target_y), voxel_size, voxel_size,
                      facecolor='yellow', edgecolor='red', linewidth=3, alpha=0.3)
ax1.add_patch(highlight)
ax1.text(target_x + 0.05, target_y + 0.08, 'ALL 5 POINTS\n→ SAME VOXEL',
        ha='center', va='center', fontsize=10, weight='bold', color='red',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                 edgecolor='red', linewidth=2))

# ============================================================================
# SUBPLOT 2: Mathematical Operation
# ============================================================================
ax2 = plt.subplot(2, 3, 3)
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.text(5, 9, 'Voxel Index Computation', ha='center', va='top',
        fontsize=14, weight='bold')

# Show the formula
formula_text = r'$\mathbf{Voxel\ Index} = \left\lfloor \frac{\mathbf{Point\ Position}}{\mathbf{Voxel\ Size}} \right\rfloor$'
ax2.text(5, 7.5, formula_text, ha='center', va='center', fontsize=16,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', 
                 edgecolor='#333', linewidth=2))

# Example calculation
calc_y = 5.5
examples = [
    ('Point P₁:', '(5.23, 3.67, 1.42)'),
    ('Voxel size:', 'σ = 0.10 m'),
    ('', ''),
    ('Calculation:', ''),
    ('  i_x = ⌊5.23/0.10⌋', '= ⌊52.3⌋ = 52'),
    ('  i_y = ⌊3.67/0.10⌋', '= ⌊36.7⌋ = 36'),
    ('  i_z = ⌊1.42/0.10⌋', '= ⌊14.2⌋ = 14'),
    ('', ''),
    ('Result:', 'Voxel Index = (52, 36, 14)'),
]

for i, (left, right) in enumerate(examples):
    y_pos = calc_y - i * 0.5
    ax2.text(1, y_pos, left, ha='left', va='center', fontsize=10, family='monospace')
    if right:
        weight = 'bold' if 'Result' in left or 'Voxel Index' in right else 'normal'
        ax2.text(6, y_pos, right, ha='left', va='center', fontsize=10, 
                family='monospace', weight=weight)

# ============================================================================
# SUBPLOT 3: Multi-Scale Comparison (Bottom Left)
# ============================================================================
ax3 = plt.subplot(2, 3, 4)
ax3.set_xlim(5.0, 5.4)
ax3.set_ylim(3.5, 3.9)
ax3.set_xlabel('X coordinate (meters)', fontsize=10, weight='bold')
ax3.set_ylabel('Y coordinate (meters)', fontsize=10, weight='bold')
ax3.set_title('Fine Scale (σ = 0.05m)\n5 points → 4 different voxels', 
              fontsize=11, weight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_aspect('equal')

# Fine grid
fine_size = 0.05
for x in np.arange(5.0, 5.4, fine_size):
    for y in np.arange(3.5, 3.9, fine_size):
        rect = Rectangle((x, y), fine_size, fine_size,
                        facecolor='#E6F3E6', edgecolor='#666', linewidth=0.5)
        ax3.add_patch(rect)

# Plot points with different voxel assignments
fine_assignments = {
    (5.23, 3.67): (104, 73, 'P₁', 'red'),
    (5.21, 3.65): (104, 73, 'P₂', 'red'),
    (5.28, 3.69): (105, 73, 'P₃', 'blue'),
    (5.25, 3.62): (105, 72, 'P₄', 'green'),
    (5.29, 3.68): (105, 73, 'P₅', 'blue'),
}

for (x, y), (idx_x, idx_y, label, color) in fine_assignments.items():
    ax3.plot(x, y, 'o', markersize=8, color=color, zorder=5)
    ax3.text(x, y + 0.012, label, ha='center', va='bottom', fontsize=7, weight='bold')
    
    # Highlight voxel
    voxel_x = idx_x * fine_size
    voxel_y = idx_y * fine_size
    highlight = Rectangle((voxel_x, voxel_y), fine_size, fine_size,
                          facecolor=color, edgecolor=color, linewidth=2, alpha=0.2)
    ax3.add_patch(highlight)

# ============================================================================
# SUBPLOT 4: Coarse Scale (Bottom Middle)
# ============================================================================
ax4 = plt.subplot(2, 3, 5)
ax4.set_xlim(5.0, 5.4)
ax4.set_ylim(3.5, 3.9)
ax4.set_xlabel('X coordinate (meters)', fontsize=10, weight='bold')
ax4.set_ylabel('Y coordinate (meters)', fontsize=10, weight='bold')
ax4.set_title('Coarse Scale (σ = 0.20m)\n5 points → 1 voxel', 
              fontsize=11, weight='bold', pad=10)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_aspect('equal')

# Coarse grid
coarse_size = 0.20
for x in np.arange(5.0, 5.4, coarse_size):
    for y in np.arange(3.5, 3.9, coarse_size):
        rect = Rectangle((x, y), coarse_size, coarse_size,
                        facecolor='#FFE6F0', edgecolor='#333', linewidth=1.5)
        ax4.add_patch(rect)

# Plot all points in same voxel
for x, y, _, label in example_points:
    ax4.plot(x, y, 'ro', markersize=8, zorder=5)
    ax4.text(x, y + 0.012, label, ha='center', va='bottom', fontsize=7, weight='bold')

# Highlight the single voxel containing all points
voxel_x = 26 * coarse_size
voxel_y = 18 * coarse_size
highlight = Rectangle((voxel_x, voxel_y), coarse_size, coarse_size,
                      facecolor='yellow', edgecolor='red', linewidth=3, alpha=0.3)
ax4.add_patch(highlight)
ax4.text(voxel_x + 0.1, voxel_y + 0.1, 'ALL 5\nPOINTS',
        ha='center', va='center', fontsize=9, weight='bold', color='red')

# ============================================================================
# SUBPLOT 5: Generalization Summary (Bottom Right)
# ============================================================================
ax5 = plt.subplot(2, 3, 6)
ax5.axis('off')
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.text(5, 9.5, 'How Voxelization Generalizes', ha='center', va='top',
        fontsize=13, weight='bold')

# Summary boxes
summary_data = [
    ('Fine Scale (σ=0.05m)', 
     ['5 points → 4 voxels', 'More detail', 'More sparse', 'Better boundaries'],
     '#E6F3E6'),
    ('Medium Scale (σ=0.10m)',
     ['5 points → 1 voxel', 'Balanced', 'Average density', 'Standard choice'],
     '#E8F4F8'),
    ('Coarse Scale (σ=0.20m)',
     ['5 points → 1 voxel', 'Less detail', 'More dense', 'Efficient'],
     '#FFE6F0'),
]

y_start = 8.0
for i, (title, points, color) in enumerate(summary_data):
    y_pos = y_start - i * 2.5
    
    # Title box
    box = FancyBboxPatch((0.5, y_pos - 0.3), 9, 0.5,
                         boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='#333', linewidth=2)
    ax5.add_patch(box)
    ax5.text(5, y_pos + 0.05, title, ha='center', va='center',
            fontsize=11, weight='bold')
    
    # Points
    for j, point in enumerate(points):
        ax5.text(1, y_pos - 0.6 - j*0.35, f'• {point}', ha='left', va='center',
                fontsize=9)

# Add key insight
ax5.text(5, 0.8, 'Key Insight: Voxel size controls the level of generalization', 
        ha='center', va='center', fontsize=10, style='italic', weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', 
                 edgecolor='#FFD700', linewidth=2))
ax5.text(5, 0.2, 'VoxAdapt learns optimal scale per-point instead of using fixed scale!', 
        ha='center', va='center', fontsize=9, style='italic', color='#D00',
        weight='bold')

# ============================================================================
# Main title
# ============================================================================
fig.suptitle('Voxel Indexing: How Floor Division Maps Continuous Space to Discrete Grid',
            fontsize=16, weight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
plt.savefig('voxel_indexing_explained.pdf', dpi=300, bbox_inches='tight')
plt.savefig('voxel_indexing_explained.png', dpi=300, bbox_inches='tight')
print("✅ Voxel indexing visualization saved!")
print("   - voxel_indexing_explained.pdf")
print("   - voxel_indexing_explained.png")

plt.show()
