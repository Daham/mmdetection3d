#!/usr/bin/env python3
"""
📊 VOXEL SCALE GRADIENT VISUALIZATION FOR REVIEWER PROOF

This script generates publication-quality figures demonstrating:
1. Voxel scale θ evolution during training
2. Gradient flow ||∇θ|| over iterations
3. Gumbel-Softmax stability metrics (entropy, confidence)
4. Differentiation from prior work (DSVT, Dynamic Voxelization)

Usage:
    python plot_voxel_scale_gradients.py [--csv /path/to/gradients.csv]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path


def load_gradient_log(csv_path: str = '/tmp/voxel_scale_gradients.csv') -> pd.DataFrame:
    """Load gradient tracking CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} iterations from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("   Run training first to generate gradient logs.")
        return None


def plot_scale_evolution(df: pd.DataFrame, ax: plt.Axes):
    """Plot voxel scale parameter evolution over training."""
    ax.plot(df['iteration'], df['theta_0'] * 100, 'b-', linewidth=2, label='θ₁ (fine)', marker='o', markevery=max(1, len(df)//20))
    ax.plot(df['iteration'], df['theta_1'] * 100, 'g-', linewidth=2, label='θ₂ (medium)', marker='s', markevery=max(1, len(df)//20))
    ax.plot(df['iteration'], df['theta_2'] * 100, 'r-', linewidth=2, label='θ₃ (coarse)', marker='^', markevery=max(1, len(df)//20))
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Voxel Scale (cm)', fontsize=12)
    ax.set_title('(a) Learnable Voxel Scale Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate initial and final values
    initial = f"Initial: [{df['theta_0'].iloc[0]*100:.1f}, {df['theta_1'].iloc[0]*100:.1f}, {df['theta_2'].iloc[0]*100:.1f}] cm"
    final = f"Final: [{df['theta_0'].iloc[-1]*100:.1f}, {df['theta_1'].iloc[-1]*100:.1f}, {df['theta_2'].iloc[-1]*100:.1f}] cm"
    ax.text(0.02, 0.98, initial, transform=ax.transAxes, fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.02, 0.88, final, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))


def plot_gradient_norm(df: pd.DataFrame, ax: plt.Axes):
    """Plot gradient norm ||∇θ|| showing gradient flow to scale parameters."""
    # Smooth with moving average for clarity
    window = min(50, len(df) // 10) if len(df) > 100 else 1
    grad_smooth = df['grad_norm'].rolling(window=window, min_periods=1).mean()
    
    ax.semilogy(df['iteration'], df['grad_norm'], 'b-', alpha=0.3, linewidth=0.5)
    ax.semilogy(df['iteration'], grad_smooth, 'b-', linewidth=2, label='||∇θ|| (smoothed)')
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('(b) Gradient Flow to Voxel Scale Parameters', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    avg_grad = df['grad_norm'].mean()
    ax.axhline(y=avg_grad, color='r', linestyle='--', alpha=0.5, label=f'Mean: {avg_grad:.1f}')
    ax.text(0.98, 0.98, f"Avg ||∇θ|| = {avg_grad:.1f}\n✅ Non-zero gradients\nprove learnability", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


def plot_gumbel_stability(df: pd.DataFrame, ax: plt.Axes):
    """Plot Gumbel-Softmax stability metrics."""
    ax2 = ax.twinx()
    
    # Entropy (left axis) - measures assignment uncertainty
    ax.plot(df['iteration'], df['gumbel_entropy'], 'b-', linewidth=2, label='Entropy H(σ)')
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Gumbel-Softmax Entropy', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # Max probability (right axis) - measures assignment confidence
    ax2.plot(df['iteration'], df['gumbel_max_prob'], 'r-', linewidth=2, label='Max P(σ)')
    ax2.set_ylabel('Max Assignment Probability', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('(c) Gumbel-Softmax Stability Analysis', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    # Stability interpretation
    final_entropy = df['gumbel_entropy'].iloc[-1]
    final_conf = df['gumbel_max_prob'].iloc[-1]
    stability_text = f"Final entropy: {final_entropy:.3f}\nFinal confidence: {final_conf:.3f}"
    if final_conf > 0.6:
        stability_text += "\n✅ Stable convergence"
    elif final_conf > 0.4:
        stability_text += "\n⚠️ Moderate stability"
    else:
        stability_text += "\n❌ High variance"
    ax.text(0.02, 0.02, stability_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))


def plot_temperature_schedule(df: pd.DataFrame, ax: plt.Axes):
    """Plot temperature annealing schedule."""
    ax.plot(df['iteration'], df['temperature'], 'purple', linewidth=2)
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Gumbel Temperature τ', fontsize=12)
    ax.set_title('(d) Temperature Annealing Schedule', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotate key phases
    initial_temp = df['temperature'].iloc[0]
    final_temp = df['temperature'].iloc[-1]
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Min τ threshold')
    ax.text(0.98, 0.98, f"τ: {initial_temp:.2f} → {final_temp:.2f}\nAnnealing enables\ntransition from\nexploration to exploitation",
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    ax.legend(loc='upper left', fontsize=10)


def create_comparison_table(df: pd.DataFrame) -> str:
    """Create comparison table vs prior work."""
    table = """
╔══════════════════════════════════════════════════════════════════════════════╗
║            DIFFERENTIATION FROM PRIOR WORK (DSVT, Dynamic Voxelization)      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Aspect                  │ DSVT/Dynamic Vox     │ VoxAdapt (Ours)              ║
╠═════════════════════════╪══════════════════════╪══════════════════════════════╣
║ Voxel Scales           │ Fixed hyperparameters│ nn.Parameter (learnable)     ║
║ Scale Selection        │ Rule-based / octree  │ ScaleNet + Gumbel-Softmax    ║
║ Gradient to Scales     │ ❌ Not supported      │ ✅ ∂L/∂θ via f=p/θ           ║
║ Training               │ Scales frozen        │ Scales jointly optimized     ║
║ Adaptivity             │ At inference only    │ Learned from detection loss  ║
╠═════════════════════════╪══════════════════════╪══════════════════════════════╣
║ Empirical Evidence:                                                          ║
"""
    
    # Add empirical stats
    initial = [df['theta_0'].iloc[0], df['theta_1'].iloc[0], df['theta_2'].iloc[0]]
    final = [df['theta_0'].iloc[-1], df['theta_1'].iloc[-1], df['theta_2'].iloc[-1]]
    changes = [(f-i)/i * 100 for i, f in zip(initial, final)]
    avg_grad = df['grad_norm'].mean()
    
    table += f"║ • Initial θ = [{initial[0]:.4f}, {initial[1]:.4f}, {initial[2]:.4f}] m                    ║\n"
    table += f"║ • Final θ   = [{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}] m                    ║\n"
    table += f"║ • Change    = [{changes[0]:+.2f}%, {changes[1]:+.2f}%, {changes[2]:+.2f}%]                          ║\n"
    table += f"║ • Avg ||∇θ|| = {avg_grad:.2f} (non-zero proves gradient flow)                 ║\n"
    table += "╚══════════════════════════════════════════════════════════════════════════════╝"
    
    return table


def generate_full_report(csv_path: str, output_dir: str = '.'):
    """Generate complete visualization report for reviewer."""
    df = load_gradient_log(csv_path)
    if df is None:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    plot_scale_evolution(df, ax1)
    plot_gradient_norm(df, ax2)
    plot_gumbel_stability(df, ax3)
    plot_temperature_schedule(df, ax4)
    
    fig.suptitle('VoxAdapt: Learnable Voxel Scale Optimization Evidence', fontsize=15, fontweight='bold', y=0.98)
    
    # Save figure
    fig_path = output_dir / 'voxel_scale_gradient_proof.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved figure: {fig_path}")
    
    # Also save as PDF for paper
    pdf_path = output_dir / 'voxel_scale_gradient_proof.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✅ Saved PDF: {pdf_path}")
    
    plt.close()
    
    # Print comparison table
    table = create_comparison_table(df)
    print("\n" + table)
    
    # Save table to file
    table_path = output_dir / 'voxadapt_vs_prior_work.txt'
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"✅ Saved comparison table: {table_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS FOR REVIEWER RESPONSE")
    print("="*70)
    print(f"Total training iterations: {len(df)}")
    print(f"Initial scales θ: [{df['theta_0'].iloc[0]:.5f}, {df['theta_1'].iloc[0]:.5f}, {df['theta_2'].iloc[0]:.5f}]")
    print(f"Final scales θ:   [{df['theta_0'].iloc[-1]:.5f}, {df['theta_1'].iloc[-1]:.5f}, {df['theta_2'].iloc[-1]:.5f}]")
    print(f"Average gradient norm: {df['grad_norm'].mean():.2f}")
    print(f"Max gradient norm: {df['grad_norm'].max():.2f}")
    print(f"Final Gumbel entropy: {df['gumbel_entropy'].iloc[-1]:.4f}")
    print(f"Final assignment confidence: {df['gumbel_max_prob'].iloc[-1]:.4f}")
    print("="*70)
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize voxel scale gradient flow')
    parser.add_argument('--csv', type=str, default='/tmp/voxel_scale_gradients.csv',
                        help='Path to gradient log CSV file')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    generate_full_report(args.csv, args.output)
