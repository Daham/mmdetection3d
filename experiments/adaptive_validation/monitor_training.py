#!/usr/bin/env python3
"""
Monitor adaptive training progress and parameter evolution.
"""
import json
import time
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """Parse training log file to extract metrics."""
    metrics = {
        'iterations': [],
        'loss': [],
        'lr': [],
        'adaptive_sizes': [],
        'memory_usage': []
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'INFO' in line and 'Epoch' in line:
                # Parse iteration and loss
                iter_match = re.search(r'iter: (\d+)', line)
                loss_match = re.search(r'loss: ([\d\.]+)', line)
                lr_match = re.search(r'lr: ([\d\.e-]+)', line)
                
                if iter_match:
                    metrics['iterations'].append(int(iter_match.group(1)))
                if loss_match and len(metrics['iterations']) > len(metrics['loss']):
                    metrics['loss'].append(float(loss_match.group(1)))
                if lr_match and len(metrics['iterations']) > len(metrics['lr']):
                    metrics['lr'].append(float(lr_match.group(1)))
            
            # Parse adaptive parameters if present
            if 'adaptive_sizes' in line:
                size_match = re.search(r'adaptive_sizes: \[([\d\., ]+)\]', line)
                if size_match:
                    sizes = [float(x.strip()) for x in size_match.group(1).split(',')]
                    metrics['adaptive_sizes'].append(sizes)
    
    return metrics

def plot_comparison(baseline_metrics, adaptive_metrics, save_path):
    """Plot comparison between baseline and adaptive training."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline vs Adaptive SECOND Training Comparison')
    
    # Loss comparison
    axes[0, 0].plot(baseline_metrics['iterations'], baseline_metrics['loss'], 
                   label='Baseline SECOND', color='blue')
    axes[0, 0].plot(adaptive_metrics['iterations'], adaptive_metrics['loss'], 
                   label='Adaptive SECOND', color='red')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Learning rate
    axes[0, 1].plot(baseline_metrics['iterations'], baseline_metrics['lr'], 
                   label='Baseline', color='blue')
    axes[0, 1].plot(adaptive_metrics['iterations'], adaptive_metrics['lr'], 
                   label='Adaptive', color='red')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    
    # Adaptive parameter evolution
    if adaptive_metrics['adaptive_sizes']:
        adaptive_sizes = adaptive_metrics['adaptive_sizes']
        iterations = list(range(len(adaptive_sizes)))
        
        # Plot x, y, z adaptive sizes
        for i, dim in enumerate(['X', 'Y', 'Z']):
            dim_sizes = [sizes[i] if len(sizes) > i else 1.0 for sizes in adaptive_sizes]
            axes[1, 0].plot(iterations, dim_sizes, label=f'{dim} size factor')
        
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Size Factor')
        axes[1, 0].set_title('Adaptive Voxel Size Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No adaptive size data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Adaptive Sizes (No Data)')
    
    # Performance summary
    if baseline_metrics['loss'] and adaptive_metrics['loss']:
        final_baseline_loss = baseline_metrics['loss'][-1]
        final_adaptive_loss = adaptive_metrics['loss'][-1]
        
        improvement = ((final_baseline_loss - final_adaptive_loss) / final_baseline_loss) * 100
        
        summary_text = f"""
Training Summary:
Baseline Final Loss: {final_baseline_loss:.4f}
Adaptive Final Loss: {final_adaptive_loss:.4f}
Improvement: {improvement:.2f}%

Adaptive Benefits:
✓ Dynamic voxel adaptation
✓ Content-aware processing  
✓ Density-based optimization
"""
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")

def monitor_training(experiment_dir):
    """Monitor training progress and generate reports."""
    
    baseline_log = experiment_dir / "baseline" / "*.log"
    adaptive_log = experiment_dir / "adaptive" / "*.log"
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring and generate final report")
    
    try:
        while True:
            # Parse current logs
            baseline_metrics = parse_log_file(baseline_log)
            adaptive_metrics = parse_log_file(adaptive_log)
            
            # Print current status
            print(f"\rBaseline iterations: {len(baseline_metrics['iterations'])}, "
                  f"Adaptive iterations: {len(adaptive_metrics['iterations'])}", end='')
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nGenerating final comparison report...")
        
        # Generate final plots
        plot_path = experiment_dir / "results" / "training_comparison.png"
        plot_comparison(baseline_metrics, adaptive_metrics, plot_path)
        
        # Save metrics
        results = {
            'baseline': baseline_metrics,
            'adaptive': adaptive_metrics,
            'timestamp': time.time()
        }
        
        results_path = experiment_dir / "results" / "training_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    import sys
    experiment_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/adaptive_validation")
    monitor_training(experiment_dir)
