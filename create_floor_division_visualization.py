#!/usr/bin/env python3
"""
Create a detailed step-by-step illustration of the floor division operation
Shows exactly how continuous coordinates become discrete voxel indices
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Floor Division in Voxelization: Step-by-Step Mathematical Process',
            fontsize=16, weight='bold', y=0.98)

# ============================================================================
# SUBPLOT 1: Number Line Visualization
# ============================================================================
ax1 = axes[0, 0]
ax1.set_xlim(5.0, 5.5)
ax1.set_ylim(-0.5, 2.5)
ax1.set_xlabel('X coordinate (meters)', fontsize=12, weight='bold')
ax1.set_title('Step 1: Division (Point Position / Voxel Size)', 
             fontsize=13, weight='bold', pad=15)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_yticks([])

# Draw continuous number line
ax1.plot([5.0, 5.5], [1, 1], 'k-', linewidth=2)

# Mark voxel boundaries
voxel_size = 0.10
for x in np.arange(5.0, 5.6, voxel_size):
    ax1.plot([x, x], [0.9, 1.1], 'k-', linewidth=2)
    ax1.text(x, 0.7, f'{x:.2f}m', ha='center', va='top', fontsize=9)
    
    # Show voxel index
    idx = int(x / voxel_size)
    ax1.text(x, 1.3, f'idx={idx}', ha='center', va='bottom', 
            fontsize=8, color='blue', weight='bold')

# Mark example point
point_x = 5.23
ax1.plot(point_x, 1, 'ro', markersize=15, zorder=5)
ax1.text(point_x, 0.3, f'Point: {point_x}m', ha='center', va='top', 
        fontsize=11, weight='bold', color='red')

# Show division result
division_result = point_x / voxel_size
ax1.text(point_x, 1.8, f'{point_x} ÷ {voxel_size} = {division_result:.1f}', 
        ha='center', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                 edgecolor='orange', linewidth=2))

# ============================================================================
# SUBPLOT 2: Floor Operation
# ============================================================================
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Step 2: Floor Function (⌊52.3⌋)', 
             fontsize=13, weight='bold', pad=15)

# Visualization of floor function
y_pos = 8
box_continuous = FancyBboxPatch((1, y_pos), 3, 1.5,
                                boxstyle="round,pad=0.2",
                                facecolor='#FFE6F0', edgecolor='#333', linewidth=2)
ax2.add_patch(box_continuous)
ax2.text(2.5, y_pos + 0.75, 'Continuous\nValue', ha='center', va='center',
        fontsize=11, weight='bold')
ax2.text(2.5, y_pos + 0.2, '52.3', ha='center', va='center',
        fontsize=16, family='monospace', weight='bold', color='red')

# Arrow
arrow = FancyArrowPatch((4, y_pos + 0.75), (6, y_pos + 0.75),
                       arrowstyle='->', mutation_scale=30, linewidth=3, color='#333')
ax2.add_patch(arrow)
ax2.text(5, y_pos + 1.3, '⌊ ⌋', ha='center', va='center',
        fontsize=20, weight='bold')

box_discrete = FancyBboxPatch((6, y_pos), 3, 1.5,
                             boxstyle="round,pad=0.2",
                             facecolor='#E6F3E6', edgecolor='#333', linewidth=2)
ax2.add_patch(box_discrete)
ax2.text(7.5, y_pos + 0.75, 'Discrete\nIndex', ha='center', va='center',
        fontsize=11, weight='bold')
ax2.text(7.5, y_pos + 0.2, '52', ha='center', va='center',
        fontsize=16, family='monospace', weight='bold', color='blue')

# Explanation
explanation = [
    'Floor function ⌊x⌋ rounds DOWN to nearest integer:',
    '',
    '⌊52.3⌋ = 52  (not 52.3)',
    '⌊52.7⌋ = 52  (not 53)',
    '⌊52.0⌋ = 52',
    '⌊52.9⌋ = 52',
    '',
    'All values in [52.0, 53.0) → index 52',
]

y_text = 6
for line in explanation:
    weight = 'bold' if '⌊' in line or 'All values' in line else 'normal'
    size = 10 if weight == 'bold' else 9
    ax2.text(5, y_text, line, ha='center', va='center', fontsize=size, 
            weight=weight, family='monospace' if '⌊' in line else 'sans-serif')
    y_text -= 0.5

# Visual number line showing range
ax2.plot([2, 8], [2, 2], 'b-', linewidth=3)
ax2.plot([2, 2], [1.8, 2.2], 'b-', linewidth=3)
ax2.plot([8, 8], [1.8, 2.2], 'b-', linewidth=3)
ax2.text(2, 1.5, '52.0', ha='center', va='top', fontsize=10, weight='bold')
ax2.text(8, 1.5, '53.0', ha='center', va='top', fontsize=10, weight='bold')
ax2.text(5, 2.5, 'Range [52.0, 53.0) → Voxel Index 52', 
        ha='center', va='bottom', fontsize=11, weight='bold', color='blue')

# ============================================================================
# SUBPLOT 3: 3D Example
# ============================================================================
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Step 3: Apply to All 3 Dimensions (X, Y, Z)', 
             fontsize=13, weight='bold', pad=15)

# Show calculation for 3D point
calc_data = [
    ('Point Position:', 'P = (5.23, 3.67, 1.42) meters', '#FFE6E6'),
    ('Voxel Size:', 'σ = 0.10 meters', '#E8F4F8'),
    ('', '', None),
    ('X dimension:', '', None),
    ('  5.23 ÷ 0.10 = 52.3', '  ⌊52.3⌋ = 52', '#FFE6F0'),
    ('', '', None),
    ('Y dimension:', '', None),
    ('  3.67 ÷ 0.10 = 36.7', '  ⌊36.7⌋ = 36', '#FFE6F0'),
    ('', '', None),
    ('Z dimension:', '', None),
    ('  1.42 ÷ 0.10 = 14.2', '  ⌊14.2⌋ = 14', '#FFE6F0'),
    ('', '', None),
    ('Voxel Index:', 'i = (52, 36, 14)', '#E6F3E6'),
]

y_pos = 9
for left, right, bgcolor in calc_data:
    if not left and not right:
        y_pos -= 0.4
        continue
    
    size = 11 if 'Point' in left or 'Voxel' in left else 10
    weight = 'bold' if 'Point' in left or 'Voxel' in left or 'dimension' in left else 'normal'
    family = 'monospace' if '÷' in left or '⌊' in left or '=' in right else 'sans-serif'
    
    if bgcolor:
        box = FancyBboxPatch((0.5, y_pos - 0.15), 9, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor=bgcolor, edgecolor='#333', linewidth=1.5)
        ax3.add_patch(box)
    
    ax3.text(1, y_pos, left, ha='left', va='center', fontsize=size, 
            weight=weight, family=family)
    if right:
        ax3.text(5.5, y_pos, right, ha='left', va='center', fontsize=size, 
                weight=weight, family=family)
    
    y_pos -= 0.6

# Add visual representation
ax3.text(5, 1.5, '📦 Voxel (52, 36, 14) contains all points in region:', 
        ha='center', va='center', fontsize=10, weight='bold')
ax3.text(5, 0.8, 'X: [5.20, 5.30) m  •  Y: [3.60, 3.70) m  •  Z: [1.40, 1.50) m',
        ha='center', va='center', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', 
                 edgecolor='#FFD700', linewidth=2))

# ============================================================================
# SUBPLOT 4: Many-to-One Mapping
# ============================================================================
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Result: Many-to-One Generalization', 
             fontsize=13, weight='bold', pad=15)

# Left side: Multiple points
y_points = 8
points_data = [
    ('P₁', '(5.23, 3.67, 1.42)', '→ (52, 36, 14)'),
    ('P₂', '(5.21, 3.65, 1.45)', '→ (52, 36, 14)'),
    ('P₃', '(5.28, 3.69, 1.41)', '→ (52, 36, 14)'),
    ('P₄', '(5.25, 3.62, 1.48)', '→ (52, 36, 14)'),
    ('P₅', '(5.29, 3.68, 1.43)', '→ (52, 36, 14)'),
]

for i, (label, coords, result) in enumerate(points_data):
    y = y_points - i * 0.7
    
    # Point circle
    circle = plt.Circle((1, y), 0.2, color='red', zorder=3)
    ax4.add_patch(circle)
    ax4.text(1, y, label, ha='center', va='center', fontsize=9, 
            weight='bold', color='white')
    
    # Coordinates
    ax4.text(2, y, coords, ha='left', va='center', fontsize=9, 
            family='monospace')
    
    # Arrow
    arrow = FancyArrowPatch((4.5, y), (6.5, y),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=2, color='#666')
    ax4.add_patch(arrow)
    
    # Result
    ax4.text(7, y, result, ha='left', va='center', fontsize=9,
            family='monospace', color='blue', weight='bold')

# Voxel box on right
voxel_box = FancyBboxPatch((7, 4.5), 2.5, 4,
                          boxstyle="round,pad=0.2",
                          facecolor='#E6F3E6', edgecolor='blue', linewidth=3)
ax4.add_patch(voxel_box)
ax4.text(8.25, 6.5, 'Voxel\n(52,36,14)', ha='center', va='center',
        fontsize=12, weight='bold', color='blue')
ax4.text(8.25, 5.5, '5 points\naggregated', ha='center', va='center',
        fontsize=9, style='italic')

# Bottom summary
summary_box = FancyBboxPatch((0.5, 0.5), 9, 2.5,
                            boxstyle="round,pad=0.3",
                            facecolor='#FFFACD', edgecolor='#FFD700', linewidth=2)
ax4.add_patch(summary_box)
ax4.text(5, 2.5, 'Key Insight: Generalization via Floor Division', 
        ha='center', va='center', fontsize=12, weight='bold')
ax4.text(5, 1.9, '• Infinite continuous positions → Finite discrete indices', 
        ha='center', va='center', fontsize=10)
ax4.text(5, 1.4, '• All points in same 10cm³ cube → Same voxel index', 
        ha='center', va='center', fontsize=10)
ax4.text(5, 0.9, '• Reduces complexity but loses fine-grained position info', 
        ha='center', va='center', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
plt.savefig('floor_division_voxelization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('floor_division_voxelization.png', dpi=300, bbox_inches='tight')
print("✅ Floor division visualization saved!")
print("   - floor_division_voxelization.pdf")
print("   - floor_division_voxelization.png")

plt.show()
