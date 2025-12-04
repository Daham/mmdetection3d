"""
Generate a detailed technical diagram explaining ScaleNet's MLP, Gumbel-Softmax, and learnable scales.
This diagram focuses on the mathematical operations and data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

def create_technical_scalenet_diagram():
    """Create detailed technical diagram of ScaleNet architecture"""
    
    # Create large figure for detailed explanation
    fig = plt.figure(figsize=(28, 20), dpi=300)
    ax = plt.subplot(111)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Professional color scheme
    colors = {
        'input': '#E3F2FD',
        'learnable': '#FFEBEE',
        'mlp': '#E8F5E9',
        'gumbel': '#FFF3E0',
        'voxel': '#F3E5F5',
        'feature': '#FFF9C4',
        'highlight': '#FF5252',
        'param': '#FF6B6B',
        'arrow': '#263238'
    }
    
    # Main title
    fig.suptitle('ScaleNet: Detailed Technical Architecture\nLearnable Multi-Scale Voxelization with MLP and Gumbel-Softmax',
                 fontsize=22, fontweight='bold', y=0.98)
    
    # ============================================================
    # PART 1: INPUT & LEARNABLE PARAMETERS (Top Section)
    # ============================================================
    
    y_top = 17
    
    # Section 1A: Input Point Cloud
    ax.text(4, y_top, '1. INPUT POINT CLOUD', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['input'], edgecolor='black', linewidth=2))
    
    input_box = FancyBboxPatch((1.5, y_top - 2.5), 5, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    
    ax.text(4, y_top - 0.8, r'Point Cloud: $P = \{p_i\}_{i=1}^{N}$', fontsize=11, ha='center', fontweight='bold')
    ax.text(4, y_top - 1.2, r'Each point: $p_i = (x_i, y_i, z_i, r_i)$', fontsize=10, ha='center')
    ax.text(4, y_top - 1.6, r'where $(x,y,z)$ = position', fontsize=9, ha='center')
    ax.text(4, y_top - 2.0, r'$r$ = reflectance intensity', fontsize=9, ha='center')
    
    # Section 1B: Learnable Voxel Scale Parameters
    ax.text(11.5, y_top, '2. LEARNABLE VOXEL SCALES (PhD Contribution)', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['learnable'], edgecolor=colors['highlight'], linewidth=3))
    
    param_box = FancyBboxPatch((8, y_top - 2.5), 7, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['learnable'],
                               edgecolor=colors['highlight'], linewidth=3)
    ax.add_patch(param_box)
    
    ax.text(11.5, y_top - 0.7, r'$theta_scale = [\theta_1, \theta_2, \theta_3]$', 
            fontsize=12, ha='center', fontweight='bold', color=colors['param'])
    ax.text(11.5, y_top - 1.1, 'Initialization:', fontsize=10, ha='center', fontweight='bold')
    ax.text(11.5, y_top - 1.45, r'$\theta_1 = 0.05m$ (fine - small objects)', fontsize=9, ha='center')
    ax.text(11.5, y_top - 1.75, r'$\theta_2 = 0.10m$ (medium - cars)', fontsize=9, ha='center')
    ax.text(11.5, y_top - 2.05, r'$\theta_3 = 0.20m$ (coarse - large vehicles)', fontsize=9, ha='center')
    
    param_box2 = FancyBboxPatch((8, y_top - 3.8), 7, 1,
                               boxstyle="round,pad=0.08",
                               facecolor='lightyellow',
                               edgecolor='black', linewidth=1)
    ax.add_patch(param_box2)
    ax.text(11.5, y_top - 3.0, 'PyTorch Implementation:', fontsize=9, ha='center', fontweight='bold')
    ax.text(11.5, y_top - 3.35, r'self.voxel_scales = nn.Parameter(', fontsize=8, ha='center', family='monospace')
    ax.text(11.5, y_top - 3.65, r'    torch.tensor([0.05, 0.10, 0.20]), requires_grad=True)', fontsize=8, ha='center', family='monospace')
    
    # Arrow indicating these are used together
    arrow_to_mlp = FancyArrowPatch((6.5, y_top - 2.5), (11.5, y_top - 4.8),
                                   arrowstyle='->', mutation_scale=25, linewidth=3,
                                   color='black')
    ax.add_patch(arrow_to_mlp)
    ax.text(9, y_top - 3.5, 'Feed to\nScaleNet', fontsize=10, ha='center', fontweight='bold')
    
    # ============================================================
    # PART 2: MLP ARCHITECTURE (Middle-Upper Section)
    # ============================================================
    
    y_mlp = 12
    
    ax.text(14, y_mlp + 1.2, '3. SCALE ASSIGNMENT NETWORK (MLP)', fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=colors['mlp'], edgecolor='black', linewidth=3))
    
    # Detailed MLP architecture
    mlp_main_box = FancyBboxPatch((4, y_mlp - 4.5), 20, 5,
                                  boxstyle="round,pad=0.15",
                                  facecolor='white',
                                  edgecolor='black', linewidth=3)
    ax.add_patch(mlp_main_box)
    
    # Layer 1: Input
    layer1_x = 5.5
    layer1_box = FancyBboxPatch((layer1_x - 0.8, y_mlp - 3.5), 1.6, 3.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['input'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(layer1_box)
    ax.text(layer1_x, y_mlp + 0.3, 'Input Layer', fontsize=10, ha='center', fontweight='bold')
    ax.text(layer1_x, y_mlp - 0.2, r'$p_i = (x,y,z,r)$', fontsize=10, ha='center')
    ax.text(layer1_x, y_mlp - 0.7, 'Shape:', fontsize=8, ha='center', style='italic')
    ax.text(layer1_x, y_mlp - 1.1, '[N, 4]', fontsize=9, ha='center', family='monospace')
    ax.text(layer1_x, y_mlp - 1.7, r'$N$ = num points', fontsize=8, ha='center')
    ax.text(layer1_x, y_mlp - 2.2, '4 = features', fontsize=8, ha='center')
    ax.text(layer1_x, y_mlp - 2.7, '(x, y, z, r)', fontsize=8, ha='center')
    
    # Layer 2: Hidden Layer 1 (Linear + ReLU)
    layer2_x = 9
    layer2_box = FancyBboxPatch((layer2_x - 1.2, y_mlp - 3.5), 2.4, 3.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['mlp'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(layer2_box)
    ax.text(layer2_x, y_mlp + 0.3, 'Hidden Layer 1', fontsize=10, ha='center', fontweight='bold')
    ax.text(layer2_x, y_mlp - 0.2, r'Linear: $4 \to 64$', fontsize=10, ha='center')
    ax.text(layer2_x, y_mlp - 0.65, r'$h_1 = ReLU(W_1 p_i + b_1)$', fontsize=9, ha='center')
    ax.text(layer2_x, y_mlp - 1.2, 'Shape:', fontsize=8, ha='center', style='italic')
    ax.text(layer2_x, y_mlp - 1.6, '[N, 64]', fontsize=9, ha='center', family='monospace')
    ax.text(layer2_x, y_mlp - 2.2, 'Parameters:', fontsize=8, ha='center', style='italic')
    ax.text(layer2_x, y_mlp - 2.6, r'$W_1 \in \mathbb{R}^{64 \times 4}$', fontsize=8, ha='center')
    ax.text(layer2_x, y_mlp - 3.0, r'$b_1 \in \mathbb{R}^{64}$', fontsize=8, ha='center')
    
    # Arrow 1->2
    arrow_12 = FancyArrowPatch((layer1_x + 0.8, y_mlp - 1.5), (layer2_x - 1.2, y_mlp - 1.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax.add_patch(arrow_12)
    
    # Layer 3: Hidden Layer 2 (Linear + ReLU)
    layer3_x = 13.5
    layer3_box = FancyBboxPatch((layer3_x - 1.2, y_mlp - 3.5), 2.4, 3.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['mlp'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(layer3_box)
    ax.text(layer3_x, y_mlp + 0.3, 'Hidden Layer 2', fontsize=10, ha='center', fontweight='bold')
    ax.text(layer3_x, y_mlp - 0.2, r'Linear: $64 \to 32$', fontsize=10, ha='center')
    ax.text(layer3_x, y_mlp - 0.65, r'$h_2 = ReLU(W_2 h_1 + b_2)$', fontsize=9, ha='center')
    ax.text(layer3_x, y_mlp - 1.2, 'Shape:', fontsize=8, ha='center', style='italic')
    ax.text(layer3_x, y_mlp - 1.6, '[N, 32]', fontsize=9, ha='center', family='monospace')
    ax.text(layer3_x, y_mlp - 2.2, 'Parameters:', fontsize=8, ha='center', style='italic')
    ax.text(layer3_x, y_mlp - 2.6, r'$W_2 \in \mathbb{R}^{32 \times 64}$', fontsize=8, ha='center')
    ax.text(layer3_x, y_mlp - 3.0, r'$b_2 \in \mathbb{R}^{32}$', fontsize=8, ha='center')
    
    # Arrow 2->3
    arrow_23 = FancyArrowPatch((layer2_x + 1.2, y_mlp - 1.5), (layer3_x - 1.2, y_mlp - 1.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax.add_patch(arrow_23)
    
    # Layer 4: Output Layer (Linear, no activation)
    layer4_x = 18
    layer4_box = FancyBboxPatch((layer4_x - 1.2, y_mlp - 3.5), 2.4, 3.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['gumbel'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(layer4_box)
    ax.text(layer4_x, y_mlp + 0.3, 'Output Layer', fontsize=10, ha='center', fontweight='bold')
    ax.text(layer4_x, y_mlp - 0.2, r'Linear: $32 \to 3$', fontsize=10, ha='center')
    ax.text(layer4_x, y_mlp - 0.65, r'$z_i = W_3 h_2 + b_3$', fontsize=9, ha='center')
    ax.text(layer4_x, y_mlp - 1.2, 'Shape:', fontsize=8, ha='center', style='italic')
    ax.text(layer4_x, y_mlp - 1.6, '[N, 3]', fontsize=9, ha='center', family='monospace')
    ax.text(layer4_x, y_mlp - 2.1, 'Logits (unnormalized)', fontsize=8, ha='center', style='italic')
    ax.text(layer4_x, y_mlp - 2.6, r'$z_i = [z_i^{(1)}, z_i^{(2)}, z_i^{(3)}]$', fontsize=8, ha='center')
    ax.text(layer4_x, y_mlp - 3.1, 'One logit per scale', fontsize=8, ha='center')
    
    # Arrow 3->4
    arrow_34 = FancyArrowPatch((layer3_x + 1.2, y_mlp - 1.5), (layer4_x - 1.2, y_mlp - 1.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax.add_patch(arrow_34)
    
    # Layer 5: Logit visualization
    layer5_x = 22
    layer5_box = FancyBboxPatch((layer5_x - 1.5, y_mlp - 3.5), 3, 3.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lightyellow',
                                edgecolor='orange', linewidth=2)
    ax.add_patch(layer5_box)
    ax.text(layer5_x, y_mlp + 0.3, 'Logits per Point', fontsize=10, ha='center', fontweight='bold')
    ax.text(layer5_x, y_mlp - 0.3, r'Example for point $i$:', fontsize=9, ha='center', style='italic')
    ax.text(layer5_x, y_mlp - 0.8, r'$z_i^{(1)} = 2.1$ (fine)', fontsize=9, ha='center')
    ax.text(layer5_x, y_mlp - 1.2, r'$z_i^{(2)} = 0.5$ (medium)', fontsize=9, ha='center')
    ax.text(layer5_x, y_mlp - 1.6, r'$z_i^{(3)} = -1.2$ (coarse)', fontsize=9, ha='center')
    ax.text(layer5_x, y_mlp - 2.2, 'Higher logit =', fontsize=8, ha='center', style='italic')
    ax.text(layer5_x, y_mlp - 2.6, 'stronger preference', fontsize=8, ha='center', style='italic')
    ax.text(layer5_x, y_mlp - 3.0, 'for that scale', fontsize=8, ha='center', style='italic')
    
    # Arrow 4->5
    arrow_45 = FancyArrowPatch((layer4_x + 1.2, y_mlp - 1.5), (layer5_x - 1.5, y_mlp - 1.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='orange')
    ax.add_patch(arrow_45)
    
    # ============================================================
    # PART 3: GUMBEL-SOFTMAX (Middle Section)
    # ============================================================
    
    y_gumbel = 5.5
    
    ax.text(14, y_gumbel + 1.5, '4. GUMBEL-SOFTMAX: Differentiable Discrete Selection', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=colors['gumbel'], edgecolor='black', linewidth=3))
    
    # Main Gumbel-Softmax box
    gumbel_box = FancyBboxPatch((2, y_gumbel - 4), 24, 5,
                                boxstyle="round,pad=0.15",
                                facecolor='white',
                                edgecolor='black', linewidth=3)
    ax.add_patch(gumbel_box)
    
    # Step 1: Sample Gumbel noise
    step1_x = 5
    step1_box = FancyBboxPatch((step1_x - 1.8, y_gumbel - 3), 3.6, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFE0B2',
                               edgecolor='black', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(step1_x, y_gumbel + 0.3, 'Step 1: Sample Noise', fontsize=10, ha='center', fontweight='bold')
    ax.text(step1_x, y_gumbel - 0.2, r'Sample: $g_i^{(s)} \sim Gumbel(0,1)$', fontsize=9, ha='center')
    ax.text(step1_x, y_gumbel - 0.7, r'$g = -\log(-\log(u))$', fontsize=9, ha='center')
    ax.text(step1_x, y_gumbel - 1.15, r'where $u \sim Uniform(0,1)$', fontsize=8, ha='center')
    ax.text(step1_x, y_gumbel - 1.6, 'Adds stochasticity', fontsize=8, ha='center', style='italic')
    ax.text(step1_x, y_gumbel - 2.0, '(only during training)', fontsize=8, ha='center', style='italic')
    ax.text(step1_x, y_gumbel - 2.5, r'Shape: [N, 3]', fontsize=8, ha='center', family='monospace')
    
    # Step 2: Add noise to logits
    step2_x = 10
    step2_box = FancyBboxPatch((step2_x - 1.8, y_gumbel - 3), 3.6, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFE0B2',
                               edgecolor='black', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(step2_x, y_gumbel + 0.3, 'Step 2: Add to Logits', fontsize=10, ha='center', fontweight='bold')
    ax.text(step2_x, y_gumbel - 0.2, r'$\tilde{z}_i^{(s)} = z_i^{(s)} + g_i^{(s)}$', fontsize=9, ha='center')
    ax.text(step2_x, y_gumbel - 0.8, 'Example:', fontsize=8, ha='center', style='italic')
    ax.text(step2_x, y_gumbel - 1.2, r'$\tilde{z}_i = [2.3, 0.8, -0.9]$', fontsize=8, ha='center')
    ax.text(step2_x, y_gumbel - 1.7, 'Perturbed logits', fontsize=8, ha='center', style='italic')
    ax.text(step2_x, y_gumbel - 2.1, 'for exploration', fontsize=8, ha='center', style='italic')
    
    arrow_g12 = FancyArrowPatch((step1_x + 1.8, y_gumbel - 1.5), (step2_x - 1.8, y_gumbel - 1.5),
                                arrowstyle='->', mutation_scale=18, linewidth=2, color='black')
    ax.add_patch(arrow_g12)
    
    # Step 3: Apply softmax with temperature
    step3_x = 15.5
    step3_box = FancyBboxPatch((step3_x - 2.2, y_gumbel - 3), 4.4, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#C8E6C9',
                               edgecolor='black', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(step3_x, y_gumbel + 0.3, 'Step 3: Softmax + Temp', fontsize=10, ha='center', fontweight='bold')
    ax.text(step3_x, y_gumbel - 0.25, r'$\sigma_i^{(s)} = \frac{\exp(\tilde{z}_i^{(s)} / \tau)}{\sum_{s=1}^{3} \exp(\tilde{z}_i^{(s)} / \tau)}$', 
            fontsize=9, ha='center')
    ax.text(step3_x, y_gumbel - 0.85, r'Temperature $\tau = 1.0$', fontsize=8, ha='center')
    ax.text(step3_x, y_gumbel - 1.25, r'Lower $\tau$ = sharper', fontsize=8, ha='center', style='italic')
    ax.text(step3_x, y_gumbel - 1.65, r'Higher $\tau$ = softer', fontsize=8, ha='center', style='italic')
    ax.text(step3_x, y_gumbel - 2.1, r'Output: probability distribution', fontsize=8, ha='center')
    ax.text(step3_x, y_gumbel - 2.5, r'$\sum_{s=1}^{3} \sigma_i^{(s)} = 1$', fontsize=8, ha='center')
    
    arrow_g23 = FancyArrowPatch((step2_x + 1.8, y_gumbel - 1.5), (step3_x - 2.2, y_gumbel - 1.5),
                                arrowstyle='->', mutation_scale=18, linewidth=2, color='black')
    ax.add_patch(arrow_g23)
    
    # Step 4: Output probabilities
    step4_x = 22.5
    step4_box = FancyBboxPatch((step4_x - 2.2, y_gumbel - 3), 4.4, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#C8E6C9',
                               edgecolor='green', linewidth=2)
    ax.add_patch(step4_box)
    ax.text(step4_x, y_gumbel + 0.3, 'Step 4: Scale Weights', fontsize=10, ha='center', fontweight='bold')
    ax.text(step4_x, y_gumbel - 0.2, r'For each point $i$:', fontsize=9, ha='center', style='italic')
    ax.text(step4_x, y_gumbel - 0.65, r'$\sigma_i = [\sigma_i^{(1)}, \sigma_i^{(2)}, \sigma_i^{(3)}]$', fontsize=9, ha='center')
    ax.text(step4_x, y_gumbel - 1.15, 'Example:', fontsize=8, ha='center', style='italic')
    ax.text(step4_x, y_gumbel - 1.5, r'$\sigma_i = [0.73, 0.22, 0.05]$', fontsize=8, ha='center', family='monospace')
    ax.text(step4_x, y_gumbel - 1.95, r'Point $i$ prefers fine scale', fontsize=8, ha='center')
    ax.text(step4_x, y_gumbel - 2.35, r'(73% weight on $\theta_1$)', fontsize=8, ha='center')
    
    arrow_g34 = FancyArrowPatch((step3_x + 2.2, y_gumbel - 1.5), (step4_x - 2.2, y_gumbel - 1.5),
                                arrowstyle='->', mutation_scale=18, linewidth=2, color='green')
    ax.add_patch(arrow_g34)
    
    # Key property box
    key_box = FancyBboxPatch((2.5, y_gumbel - 3.8), 23, 0.6,
                             boxstyle="round,pad=0.08",
                             facecolor='lightyellow',
                             edgecolor='orange', linewidth=2)
    ax.add_patch(key_box)
    ax.text(14, y_gumbel - 3.5, 
            r'KEY: Gumbel-Softmax is DIFFERENTIABLE! Gradients flow back through softmax to MLP and $\theta_{scale}$',
            fontsize=10, ha='center', fontweight='bold', color='#D32F2F')
    
    # ============================================================
    # PART 5: HOW LEARNABLE SCALES ARE USED (Bottom Section)
    # ============================================================
    
    y_bottom = 1.5
    
    ax.text(14, y_bottom + 2.3, '5. USING LEARNABLE SCALES IN VOXELIZATION', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=colors['voxel'], edgecolor='black', linewidth=3))
    
    # Three parallel voxelization paths
    voxel_y = 0.5
    scale_positions = [6, 14, 22]
    scale_names = [r'Scale 1: $\theta_1$', r'Scale 2: $\theta_2$', r'Scale 3: $\theta_3$']
    scale_values = [r'$\approx 0.04m$', r'$\approx 0.12m$', r'$\approx 0.25m$']
    scale_colors = ['#FFCDD2', '#FFECB3', '#C5E1A5']
    weight_examples = [r'$\sigma_i^{(1)} = 0.73$', r'$\sigma_i^{(2)} = 0.22$', r'$\sigma_i^{(3)} = 0.05$']
    
    for idx, (x_pos, name, value, color, weight) in enumerate(zip(scale_positions, scale_names, scale_values, scale_colors, weight_examples)):
        # Voxelization box
        vox_box = FancyBboxPatch((x_pos - 3, voxel_y), 6, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color,
                                 edgecolor='black', linewidth=2)
        ax.add_patch(vox_box)
        
        ax.text(x_pos, voxel_y + 1.2, name, fontsize=11, ha='center', fontweight='bold')
        ax.text(x_pos, voxel_y + 0.85, f'Voxel size: {value}', fontsize=9, ha='center')
        ax.text(x_pos, voxel_y + 0.5, f'Weight: {weight}', fontsize=9, ha='center', color='green')
        ax.text(x_pos, voxel_y + 0.1, r'$\Downarrow$ Voxelize points', fontsize=8, ha='center', style='italic')
        
        # Arrow from Gumbel-Softmax output to each scale
        arrow_to_vox = FancyArrowPatch((step4_x, y_gumbel - 3), (x_pos, voxel_y + 1.5),
                                       arrowstyle='->', mutation_scale=15, linewidth=1.5,
                                       color=colors['arrow'], alpha=0.5, linestyle='dashed')
        ax.add_patch(arrow_to_vox)
    
    # Final aggregation annotation
    ax.text(14, y_bottom - 0.8, 
            r'Final feature: $f_i = \sigma_i^{(1)} \cdot f_i^{(1)} + \sigma_i^{(2)} \cdot f_i^{(2)} + \sigma_i^{(3)} \cdot f_i^{(3)}$',
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', edgecolor='blue', linewidth=2))
    
    ax.text(14, y_bottom - 1.4,
            r'where $f_i^{(s)}$ = feature extracted from scale $s$ voxelization',
            fontsize=10, ha='center', style='italic')
    
    # ============================================================
    # ANNOTATIONS: Why this works
    # ============================================================
    
    # Left sidebar: Key insights
    insights_box = FancyBboxPatch((0.3, 8), 3, 6,
                                  boxstyle="round,pad=0.15",
                                  facecolor='#FFF9C4',
                                  edgecolor='orange', linewidth=2)
    ax.add_patch(insights_box)
    
    insights_text = """WHY THIS WORKS:

1. MLP learns point-specific
   scale preferences
   
2. Gumbel-Softmax makes
   discrete selection
   differentiable
   
3. Learnable θ adapts
   to dataset/task
   
4. End-to-end training:
   gradients flow from
   detection loss back
   to voxel scales
   
5. No manual tuning!
"""
    ax.text(1.8, 13.5, insights_text, fontsize=9, ha='center', va='top',
            family='monospace', linespacing=1.6)
    
    # Right sidebar: Gradient flow
    grad_box = FancyBboxPatch((24.7, 8), 3, 6,
                              boxstyle="round,pad=0.15",
                              facecolor='#FFCDD2',
                              edgecolor='red', linewidth=2)
    ax.add_patch(grad_box)
    
    grad_text = """GRADIENT FLOW:

Detection Loss
    ↓
Feature Aggregation
    ↓
Gumbel Weights σ
    ↓
MLP Parameters
    ↓
Voxel Scales θ

All parameters
updated via
backpropagation!
"""
    ax.text(26.2, 13.5, grad_text, fontsize=9, ha='center', va='top',
            family='monospace', linespacing=1.6, color='#C62828')
    
    plt.tight_layout()
    return fig

# Generate and save
print("Creating detailed technical ScaleNet diagram...")
fig = create_technical_scalenet_diagram()

output_dir = "/home/daham/mmdetection_project/mmdetection3d"
print(f"Saving to {output_dir}/...")

# Save in multiple formats
fig.savefig(f"{output_dir}/scalenet_technical_detailed.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: scalenet_technical_detailed.png (300 DPI)")

fig.savefig(f"{output_dir}/scalenet_technical_detailed.pdf", 
            bbox_inches='tight', facecolor='white')
print("✓ Saved: scalenet_technical_detailed.pdf (vector)")

fig.savefig(f"{output_dir}/scalenet_technical_detailed.svg", 
            bbox_inches='tight', facecolor='white')
print("✓ Saved: scalenet_technical_detailed.svg (editable)")

print("\n✅ Technical ScaleNet diagram complete!")
print("   This diagram shows:")
print("   • Complete MLP architecture (4→64→32→3)")
print("   • Gumbel-Softmax mathematical operations")
print("   • How learnable scales θ are used")
print("   • Gradient flow for end-to-end training")

plt.close()
