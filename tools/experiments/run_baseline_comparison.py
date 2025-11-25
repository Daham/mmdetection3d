"""Automated baseline comparison script.

Runs all 4 baseline experiments and generates comparison tables for paper.
Supports both sequential and SLURM batch execution.

Usage:
    # Sequential execution (safe)
    python tools/experiments/run_baseline_comparison.py
    
    # Parallel SLURM execution
    python tools/experiments/run_baseline_comparison.py --use-slurm
    
    # Resume from checkpoint
    python tools/experiments/run_baseline_comparison.py --resume
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import sys

# Add mmdet3d to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class BaselineComparison:
    """Manages execution and comparison of baseline experiments."""
    
    def __init__(self, use_slurm: bool = False, resume: bool = False):
        """Initialize comparison manager.
        
        Args:
            use_slurm: Whether to use SLURM for parallel execution
            resume: Whether to resume from existing checkpoints
        """
        self.use_slurm = use_slurm
        self.resume = resume
        self.root_dir = Path(__file__).parent.parent.parent
        
        # Define experiments
        self.experiments = [
            {
                'name': 'single_scale_0.1m',
                'config': 'configs/adaptive_voxelnet/single_scale_0.1m.py',
                'description': 'Baseline: Fixed 0.1m voxels',
                'expected_ap': 65.0,
                'priority': 1
            },
            {
                'name': 'multi_scale_fixed',
                'config': 'configs/adaptive_voxelnet/multi_scale_fixed.py',
                'description': 'Naive multi-scale (no learning)',
                'expected_ap': 42.0,
                'priority': 2
            },
            {
                'name': 'multi_scale_learnable_fusion',
                'config': 'configs/adaptive_voxelnet/multi_scale_learnable_fusion.py',
                'description': 'Your previous work: Learned fusion',
                'expected_ap': 68.0,
                'priority': 3
            },
            {
                'name': 'adaptive_octree',
                'config': 'configs/adaptive_voxelnet/adaptive_octree.py',
                'description': 'This work: TRUE adaptive voxelization',
                'expected_ap': 74.0,
                'priority': 4
            }
        ]
    
    def check_prerequisites(self) -> bool:
        """Check if environment is ready for training."""
        print("🔍 Checking prerequisites...")
        
        # Check configs exist
        for exp in self.experiments:
            config_path = self.root_dir / exp['config']
            if not config_path.exists():
                print(f"❌ Config not found: {config_path}")
                return False
            print(f"✅ {exp['name']}: Config found")
        
        # Check dataset
        dataset_path = self.root_dir / 'data' / 'kitti'
        if not dataset_path.exists():
            print(f"❌ KITTI dataset not found at {dataset_path}")
            return False
        print(f"✅ KITTI dataset found")
        
        # Check CUDA
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠️  CUDA not available - training will be slow!")
            else:
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("❌ PyTorch not installed")
            return False
        
        return True
    
    def run_experiment(self, exp: Dict, gpu_id: int = 0) -> bool:
        """Run a single experiment.
        
        Args:
            exp: Experiment configuration dict
            gpu_id: GPU ID to use
            
        Returns:
            True if successful, False otherwise
        """
        config_path = self.root_dir / exp['config']
        work_dir = self.root_dir / 'work_dirs' / exp['name']
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting: {exp['name']}")
        print(f"📝 Description: {exp['description']}")
        print(f"🎯 Expected AP: {exp['expected_ap']}%")
        print(f"📂 Work dir: {work_dir}")
        print(f"{'='*80}\n")
        
        # Check if already completed
        if self.resume and (work_dir / 'epoch_20.pth').exists():
            print(f"✅ Already completed (found checkpoint). Skipping...")
            return True
        
        # Build command
        cmd = [
            'python', 'tools/train.py',
            str(config_path),
            f'--work-dir={work_dir}'
        ]
        
        if self.resume:
            cmd.append('--resume')
        
        # Set GPU
        env = {'CUDA_VISIBLE_DEVICES': str(gpu_id)}
        
        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                env={**subprocess.os.environ, **env},
                check=True,
                capture_output=False
            )
            
            elapsed = time.time() - start_time
            print(f"\n✅ Completed {exp['name']} in {elapsed/3600:.2f} hours")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed: {exp['name']}")
            print(f"Error: {e}")
            return False
    
    def submit_slurm_job(self, exp: Dict) -> str:
        """Submit experiment as SLURM job.
        
        Args:
            exp: Experiment configuration dict
            
        Returns:
            Job ID
        """
        config_path = self.root_dir / exp['config']
        work_dir = self.root_dir / 'work_dirs' / exp['name']
        
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={exp['name']}
#SBATCH --output={work_dir}/slurm_%j.out
#SBATCH --error={work_dir}/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Activate environment
source ~/mmdet_env/bin/activate

# Run training
cd {self.root_dir}
python tools/train.py {config_path} --work-dir={work_dir}
"""
        
        # Write slurm script
        script_path = work_dir / 'submit.slurm'
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(slurm_script)
        
        # Submit job
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"✅ Submitted {exp['name']} as job {job_id}")
            return job_id
        else:
            print(f"❌ Failed to submit {exp['name']}: {result.stderr}")
            return None
    
    def collect_results(self) -> List[Dict]:
        """Collect results from all experiments."""
        results = []
        
        for exp in self.experiments:
            work_dir = self.root_dir / 'work_dirs' / exp['name']
            result_file = work_dir / 'results.json'
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    exp_results = json.load(f)
                    results.append({
                        'name': exp['name'],
                        'description': exp['description'],
                        'expected_ap': exp['expected_ap'],
                        'actual_ap': exp_results.get('KITTI/Car_3d_moderate', 0.0),
                        'bev_ap': exp_results.get('KITTI/Car_bev_moderate', 0.0),
                        'inference_time': exp_results.get('inference_time_ms', 0.0)
                    })
            else:
                print(f"⚠️  Results not found for {exp['name']}")
                results.append({
                    'name': exp['name'],
                    'description': exp['description'],
                    'expected_ap': exp['expected_ap'],
                    'actual_ap': 0.0,
                    'bev_ap': 0.0,
                    'inference_time': 0.0
                })
        
        return results
    
    def generate_comparison_table(self, results: List[Dict]) -> str:
        """Generate LaTeX table for paper."""
        
        table = r"""\begin{table}[t]
\centering
\caption{Comparison of voxelization strategies on KITTI validation set.}
\label{tab:baseline_comparison}
\begin{tabular}{lcccc}
\toprule
Method & 3D AP@0.7 & BEV AP@0.7 & Inference (ms) & Improvement \\
\midrule
"""
        
        baseline_ap = results[0]['actual_ap']  # Single-scale baseline
        
        for r in results:
            improvement = r['actual_ap'] - baseline_ap
            improvement_str = f"+{improvement:.1f}\\%" if improvement > 0 else f"{improvement:.1f}\\%"
            
            table += f"{r['description']} & "
            table += f"{r['actual_ap']:.1f}\\% & "
            table += f"{r['bev_ap']:.1f}\\% & "
            table += f"{r['inference_time']:.1f} & "
            table += f"{improvement_str} \\\\\n"
        
        table += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        return table
    
    def run_all(self):
        """Run all experiments."""
        print("\n" + "="*80)
        print("🧪 ADAPTIVE VOXELIZATION BASELINE COMPARISON")
        print("="*80 + "\n")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites check failed. Please fix issues above.")
            return
        
        print("\n✅ Prerequisites check passed!")
        
        # Run experiments
        if self.use_slurm:
            print("\n📤 Submitting jobs to SLURM...")
            job_ids = []
            for exp in self.experiments:
                job_id = self.submit_slurm_job(exp)
                if job_id:
                    job_ids.append((exp['name'], job_id))
            
            print(f"\n✅ Submitted {len(job_ids)} jobs:")
            for name, job_id in job_ids:
                print(f"  - {name}: {job_id}")
            print("\nMonitor with: squeue -u $USER")
            
        else:
            print("\n🔄 Running experiments sequentially...")
            for i, exp in enumerate(self.experiments, 1):
                print(f"\n[{i}/{len(self.experiments)}] Running {exp['name']}...")
                success = self.run_experiment(exp, gpu_id=0)
                if not success:
                    print(f"\n⚠️  {exp['name']} failed. Continue anyway? (y/n)")
                    if input().lower() != 'y':
                        break
        
        print("\n" + "="*80)
        print("✅ All experiments completed!")
        print("="*80 + "\n")
        
        # Collect and display results
        print("📊 Collecting results...")
        results = self.collect_results()
        
        # Print summary table
        print("\n" + "="*80)
        print("📈 RESULTS SUMMARY")
        print("="*80)
        print(f"{'Method':<40} {'3D AP@0.7':>12} {'BEV AP@0.7':>12} {'Δ vs Baseline':>15}")
        print("-"*80)
        
        baseline_ap = results[0]['actual_ap']
        for r in results:
            improvement = r['actual_ap'] - baseline_ap
            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            print(f"{r['description']:<40} {r['actual_ap']:>11.1f}% {r['bev_ap']:>11.1f}% {improvement_str:>14}")
        
        print("="*80 + "\n")
        
        # Generate LaTeX table
        latex_table = self.generate_comparison_table(results)
        latex_path = self.root_dir / 'comparison_table.tex'
        latex_path.write_text(latex_table)
        print(f"📝 LaTeX table saved to: {latex_path}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run baseline comparison experiments')
    parser.add_argument('--use-slurm', action='store_true',
                       help='Submit jobs to SLURM instead of running sequentially')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoints')
    args = parser.parse_args()
    
    # Run comparison
    comparison = BaselineComparison(
        use_slurm=args.use_slurm,
        resume=args.resume
    )
    comparison.run_all()


if __name__ == '__main__':
    main()
