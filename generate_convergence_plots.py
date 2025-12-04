#!/usr/bin/env python3
"""
Generate training convergence plots for VoxAdapt paper.

This script extracts AP metrics from training logs and generates:
1. Training convergence curves (5 epochs)
2. Comparison across three methods: Single-Scale, Naive Multi-Scale, VoxAdapt

Output: Publication-quality figures for journal paper.
"""

import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.alpha'] = 0.3

# Color scheme for methods
COLORS = {
    'single': '#2E86AB',      # Blue
    'naive': '#A23B72',       # Purple/Magenta
    'voxadapt': '#F18F01',    # Orange
}

MARKERS = {
    'single': 'o',
    'naive': 's',
    'voxadapt': '^',
}

def extract_ap_from_log(log_path, metric='Car_3D_AP40'):
    """
    Extract AP metrics from MMDetection3D log file.
    
    Args:
        log_path: Path to log file
        metric: Which metric to extract (default: Car_3D_AP40 for moderate)
    
    Returns:
        dict: {epoch: {'easy': float, 'moderate': float, 'hard': float}}
    """
    results = {}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Process line by line, looking for validation epoch results
    current_epoch = None
    current_line_data = ""
    
    for line in lines:
        # Check if this is a validation epoch line
        epoch_match = re.search(r'Epoch\(val\)\s+\[(\d+)\]\[5001/5001\]', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_line_data = line
            
            if current_epoch not in results:
                results[current_epoch] = {}
            
            # Extract all three difficulties from this line
            for difficulty in ['easy', 'moderate', 'hard']:
                pattern = r'KITTI/' + metric + r'_' + difficulty + r'_strict:\s+([\d.]+)'
                match = re.search(pattern, line)
                if match:
                    results[current_epoch][difficulty] = float(match.group(1))
    
    return results


def plot_convergence_curves(method_data, output_dir='paper_figures', metric_name='AP@0.7 (IoU)'):
    """
    Generate convergence plot comparing three methods.
    
    Args:
        method_data: dict with structure {method_name: {epoch: {difficulty: ap_value}}}
        output_dir: Where to save figures
        metric_name: Display name for metric
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create three plots: Easy, Moderate, Hard
    difficulties = ['easy', 'moderate', 'hard']
    
    # ============= Figure 1: Moderate only (main result) =============
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    difficulty = 'moderate'
    
    for method_name, label, color, marker in [
        ('single', 'Fixed Single-Scale', COLORS['single'], MARKERS['single']),
        ('naive', 'Naive Multi-Scale', COLORS['naive'], MARKERS['naive']),
        ('voxadapt', 'VoxAdapt (Ours)', COLORS['voxadapt'], MARKERS['voxadapt'])
    ]:
        if method_name not in method_data:
            print(f"Warning: {method_name} not found in data")
            continue
        
        epochs = sorted(method_data[method_name].keys())
        ap_values = [method_data[method_name][e].get(difficulty, 0) for e in epochs]
        
        ax.plot(epochs, ap_values, 
                marker=marker, 
                markersize=7,
                linewidth=2,
                label=label,
                color=color,
                alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Car 3D AP (%) - Moderate', fontsize=11, fontweight='bold')
    ax.set_title('Training Convergence on KITTI Validation Set', fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_xlim(0.8, 5.2)
    ax.set_xticks(range(1, 6))
    
    # Add horizontal reference line at baseline final performance
    if 'single' in method_data:
        final_single = method_data['single'][5][difficulty]
        ax.axhline(y=final_single, color=COLORS['single'], linestyle=':', alpha=0.4, linewidth=1.5)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        output_path = output_dir / f'convergence_moderate.{fmt}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    # ============= Figure 2: All three difficulties =============
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    
    for idx, difficulty in enumerate(difficulties):
        ax = axes[idx]
        
        for method_name, label, color, marker in [
            ('single', 'Fixed Single-Scale', COLORS['single'], MARKERS['single']),
            ('naive', 'Naive Multi-Scale', COLORS['naive'], MARKERS['naive']),
            ('voxadapt', 'VoxAdapt (Ours)', COLORS['voxadapt'], MARKERS['voxadapt'])
        ]:
            if method_name not in method_data:
                continue
            
            epochs = sorted(method_data[method_name].keys())
            ap_values = [method_data[method_name][e].get(difficulty, 0) for e in epochs]
            
            ax.plot(epochs, ap_values, 
                    marker=marker, 
                    markersize=6,
                    linewidth=2,
                    label=label if idx == 1 else "",  # Only show legend in middle plot
                    color=color,
                    alpha=0.9)
        
        ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Car 3D AP (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{difficulty.capitalize()} Difficulty', fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_xlim(0.8, 5.2)
        ax.set_xticks(range(1, 6))
        
        # Add legend only to middle plot
        if idx == 1:
            ax.legend(loc='lower right', framealpha=0.95, fontsize=8.5)
    
    plt.suptitle('Training Convergence Across All Difficulty Levels', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        output_path = output_dir / f'convergence_all_difficulties.{fmt}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    # ============= Figure 3: Improvement over single-scale =============
    fig, ax = plt.subplots(figsize=(6, 4))
    
    difficulty = 'moderate'
    
    if 'single' in method_data and 'voxadapt' in method_data:
        epochs = sorted(method_data['single'].keys())
        
        # Calculate improvement
        single_ap = [method_data['single'][e].get(difficulty, 0) for e in epochs]
        voxadapt_ap = [method_data['voxadapt'][e].get(difficulty, 0) for e in epochs]
        improvement = [v - s for v, s in zip(voxadapt_ap, single_ap)]
        
        # Bar plot
        bars = ax.bar(epochs, improvement, 
                      color=COLORS['voxadapt'], 
                      alpha=0.8, 
                      edgecolor='black',
                      linewidth=0.8,
                      width=0.6)
        
        # Add value labels on bars
        for epoch, imp in zip(epochs, improvement):
            ax.text(epoch, imp + 0.2, f'+{imp:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('AP Improvement over Single-Scale (%)', fontsize=11, fontweight='bold')
        ax.set_title('VoxAdapt Performance Gain (Moderate Difficulty)', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(range(1, 6))
        ax.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_ylim(bottom=min(improvement) - 1, top=max(improvement) + 1.5)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        output_path = output_dir / f'improvement_over_baseline.{fmt}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def print_summary_table(method_data):
    """Print a formatted table of final results."""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY (Epoch 5)")
    print("="*70)
    print(f"{'Method':<25} {'Easy':<12} {'Moderate':<12} {'Hard':<12}")
    print("-"*70)
    
    for method_name, label in [
        ('single', 'Fixed Single-Scale'),
        ('naive', 'Naive Multi-Scale'),
        ('voxadapt', 'VoxAdapt (Ours)')
    ]:
        if method_name not in method_data or 5 not in method_data[method_name]:
            continue
        
        epoch5 = method_data[method_name][5]
        easy = epoch5.get('easy', 0)
        moderate = epoch5.get('moderate', 0)
        hard = epoch5.get('hard', 0)
        
        print(f"{label:<25} {easy:>11.2f} {moderate:>11.2f} {hard:>11.2f}")
    
    print("="*70)
    
    # Calculate improvements
    if 'single' in method_data and 'voxadapt' in method_data:
        print("\nIMPROVEMENT (VoxAdapt - Single-Scale):")
        print("-"*70)
        
        for diff in ['easy', 'moderate', 'hard']:
            single_val = method_data['single'][5].get(diff, 0)
            voxadapt_val = method_data['voxadapt'][5].get(diff, 0)
            improvement = voxadapt_val - single_val
            
            print(f"{diff.capitalize():<15} +{improvement:>6.2f} percentage points")
    
    print("="*70 + "\n")


def main():
    """Main execution function."""
    print("="*70)
    print("TRAINING CONVERGENCE PLOT GENERATOR")
    print("="*70)
    
    # Define log file paths
    base_dir = Path('work_dirs/comparison_5epochs')
    
    log_files = {
        'single': base_dir / 'method1_single' / '20251126_135551' / '20251126_135551.log',
        'naive': base_dir / 'method2_fixed' / '20251126_145331' / '20251126_145331.log',
        'voxadapt': base_dir / 'method3_learnable' / '20251126_162714' / '20251126_162714.log',
    }
    
    # Check if files exist
    print("\nChecking log files...")
    for method, path in log_files.items():
        if path.exists():
            print(f"  ✓ Found: {method} ({path.name})")
        else:
            print(f"  ✗ Missing: {method} ({path})")
    
    # Extract data from logs
    print("\nExtracting AP metrics from logs...")
    method_data = {}
    
    for method_name, log_path in log_files.items():
        if not log_path.exists():
            print(f"  Skipping {method_name} (file not found)")
            continue
        
        print(f"  Processing {method_name}...")
        results = extract_ap_from_log(log_path, metric='Car_3D_AP40')
        
        if results:
            method_data[method_name] = results
            print(f"    ✓ Extracted {len(results)} epochs")
            # Show epoch range
            epochs = sorted(results.keys())
            print(f"    Epochs: {epochs[0]} to {epochs[-1]}")
        else:
            print(f"    ✗ No data extracted")
    
    if not method_data:
        print("\n❌ Error: No data extracted from logs!")
        return
    
    # Print summary table
    print_summary_table(method_data)
    
    # Generate plots
    print("Generating convergence plots...")
    plot_convergence_curves(method_data, output_dir='paper_figures')
    
    print("\n" + "="*70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput location: paper_figures/")
    print("  • convergence_moderate.{png,pdf,svg}")
    print("  • convergence_all_difficulties.{png,pdf,svg}")
    print("  • improvement_over_baseline.{png,pdf,svg}")
    print("\nUse these figures in your journal paper results section.")
    print("="*70)


if __name__ == '__main__':
    main()
