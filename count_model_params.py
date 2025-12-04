#!/usr/bin/env python3
"""
Count model parameters for VoxAdapt vs Baseline comparison
"""

import torch
from mmdet3d.apis import init_model
from mmengine import Config
import sys

def count_parameters(model, show_details=False):
    """Count model parameters with optional detailed breakdown"""
    total_params = 0
    trainable_params = 0
    
    if show_details:
        print(f"\n{'='*80}")
        print(f"Parameter Breakdown by Module")
        print(f"{'='*80}\n")
        print(f"{'Module':<50} {'Parameters':>15} {'Trainable':>12}")
        print(f"{'-'*80}")
    
    module_params = {}
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        # Group by top-level module
        top_module = name.split('.')[0]
        if top_module not in module_params:
            module_params[top_module] = 0
        module_params[top_module] += num_params
        
        if show_details and num_params > 10000:  # Only show modules > 10K params
            trainable_str = "Yes" if param.requires_grad else "No"
            print(f"{name:<50} {num_params:>15,} {trainable_str:>12}")
    
    if show_details:
        print(f"{'-'*80}")
        print(f"{'TOTAL':<50} {total_params:>15,}")
        print(f"{'TRAINABLE':<50} {trainable_params:>15,}")
        print(f"{'='*80}\n")
        
        # Show grouped by top-level module
        print(f"\n{'='*80}")
        print(f"Parameter Count by Top-Level Module")
        print(f"{'='*80}\n")
        print(f"{'Module':<30} {'Parameters':>20} {'Percentage':>12}")
        print(f"{'-'*80}")
        for module_name, count in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_params) * 100
            print(f"{module_name:<30} {count:>20,} {pct:>11.2f}%")
        print(f"{'-'*80}")
        print(f"{'TOTAL':<30} {total_params:>20,} {100.0:>11.2f}%")
        print(f"{'='*80}\n")
    
    return total_params, trainable_params

def main():
    # Model configurations
    baseline_config = 'configs/second/baseline_01_single_scale_hardvfe.py'
    voxadapt_config = 'configs/second/baseline_03_adaptive_multiscale_learnable.py'
    
    print("="*80)
    print("VoxAdapt vs Baseline - Parameter Count Comparison")
    print("="*80)
    print("\nLoading models (this may take a moment)...\n")
    
    try:
        # Initialize models on CPU to save memory
        print("📊 Initializing Baseline model...")
        baseline_model = init_model(baseline_config, device='cpu')
        
        print("📊 Initializing VoxAdapt model...")
        voxadapt_model = init_model(voxadapt_config, device='cpu')
        
        # Count parameters
        print("\n" + "="*80)
        print("BASELINE MODEL - Detailed Analysis")
        baseline_total, baseline_trainable = count_parameters(baseline_model, show_details=True)
        
        print("\n" + "="*80)
        print("VOXADAPT MODEL - Detailed Analysis")
        voxadapt_total, voxadapt_trainable = count_parameters(voxadapt_model, show_details=True)
        
        # Calculate overhead
        overhead = voxadapt_total - baseline_total
        overhead_pct = (overhead / baseline_total) * 100
        
        # Summary
        print("\n" + "="*80)
        print("📊 COMPARISON SUMMARY")
        print("="*80)
        print(f"\nBaseline Model:")
        print(f"  Total parameters:      {baseline_total:>12,} ({baseline_total/1e6:>6.2f}M)")
        print(f"  Trainable parameters:  {baseline_trainable:>12,} ({baseline_trainable/1e6:>6.2f}M)")
        
        print(f"\nVoxAdapt Model:")
        print(f"  Total parameters:      {voxadapt_total:>12,} ({voxadapt_total/1e6:>6.2f}M)")
        print(f"  Trainable parameters:  {voxadapt_trainable:>12,} ({voxadapt_trainable/1e6:>6.2f}M)")
        
        print(f"\nOverhead:")
        print(f"  Additional parameters: {overhead:>12,} ({overhead/1e6:>6.2f}M)")
        print(f"  Percentage increase:   {overhead_pct:>12.2f}%")
        print("="*80 + "\n")
        
        # For your table
        print("✅ FOR YOUR PAPER TABLE:")
        print("="*80)
        print(f"Baseline Parameters:  {baseline_total/1e6:.1f} M")
        print(f"VoxAdapt Parameters:  {voxadapt_total/1e6:.1f} M")
        print(f"Overhead:             +{overhead_pct:.1f}%")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("  1. Config files exist at specified paths")
        print("  2. MMDetection3D is properly installed")
        print("  3. You're in the correct directory")
        sys.exit(1)

if __name__ == '__main__':
    main()
