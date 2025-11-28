#!/usr/bin/env python3
"""
Analysis Script: Validation Experiment Results

Extracts results from all 9 training runs and performs statistical analysis.

Usage:
    python tools/analysis_tools/analyze_validation_results.py \
        --work-dir work_dirs/validation_experiment \
        --output validation_results.md
"""

import argparse
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from tabulate import tabulate


def extract_results_from_log(log_file: Path) -> Dict:
    """Extract final validation results from training log"""
    results = {}
    
    if not log_file.exists():
        print(f"Warning: Log file not found: {log_file}")
        return results
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract Car 3D AP results (Easy, Moderate, Hard)
    # Pattern: KITTI/Car_3d_easy_strict: 0.7234
    pattern = r'KITTI/Car_3d_(\w+)_strict[:\s]+(\d+\.\d+)'
    matches = re.findall(pattern, content)
    
    for difficulty, ap_str in matches:
        results[f'car_3d_{difficulty}'] = float(ap_str) * 100  # Convert to percentage
    
    # Also extract BEV results
    pattern = r'KITTI/Car_bev_(\w+)_strict[:\s]+(\d+\.\d+)'
    matches = re.findall(pattern, content)
    
    for difficulty, ap_str in matches:
        results[f'car_bev_{difficulty}'] = float(ap_str) * 100
    
    return results


def collect_results(work_dir: Path) -> Dict[str, List[Dict]]:
    """Collect results from all baseline runs"""
    
    baselines = {
        'baseline_01_single_scale': [],
        'baseline_02_fixed_multiscale': [],
        'baseline_03_adaptive': []
    }
    
    for baseline_name in baselines.keys():
        # Find all seed directories for this baseline
        seed_dirs = sorted(work_dir.glob(f'{baseline_name}_seed*'))
        
        for seed_dir in seed_dirs:
            log_file = seed_dir / 'training.log'
            results = extract_results_from_log(log_file)
            
            if results:
                results['seed'] = int(seed_dir.name.split('seed')[-1])
                results['work_dir'] = str(seed_dir)
                baselines[baseline_name].append(results)
    
    return baselines


def compute_statistics(values: List[float]) -> Dict:
    """Compute mean, std, confidence interval"""
    if len(values) == 0:
        return {'mean': 0, 'std': 0, 'ci_low': 0, 'ci_high': 0}
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    
    # 95% confidence interval
    if len(values) > 1:
        ci = stats.t.interval(
            0.95, 
            len(values) - 1,
            loc=mean,
            scale=stats.sem(values)
        )
        ci_low, ci_high = ci
    else:
        ci_low, ci_high = mean, mean
    
    return {
        'mean': mean,
        'std': std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n': len(values)
    }


def perform_significance_test(values1: List[float], values2: List[float]) -> Tuple[float, float]:
    """Perform two-tailed t-test"""
    if len(values1) < 2 or len(values2) < 2:
        return 0.0, 1.0  # Can't perform test
    
    t_stat, p_value = stats.ttest_ind(values1, values2)
    return t_stat, p_value


def generate_report(results: Dict[str, List[Dict]], output_file: Path):
    """Generate markdown report with tables and analysis"""
    
    # Extract 3D AP moderate (primary metric)
    baseline_01_moderate = [r['car_3d_moderate'] for r in results['baseline_01_single_scale'] if 'car_3d_moderate' in r]
    baseline_02_moderate = [r['car_3d_moderate'] for r in results['baseline_02_fixed_multiscale'] if 'car_3d_moderate' in r]
    baseline_03_moderate = [r['car_3d_moderate'] for r in results['baseline_03_adaptive'] if 'car_3d_moderate' in r]
    
    # Compute statistics
    stats_01 = compute_statistics(baseline_01_moderate)
    stats_02 = compute_statistics(baseline_02_moderate)
    stats_03 = compute_statistics(baseline_03_moderate)
    
    # Significance tests
    t_03_vs_01, p_03_vs_01 = perform_significance_test(baseline_03_moderate, baseline_01_moderate)
    t_03_vs_02, p_03_vs_02 = perform_significance_test(baseline_03_moderate, baseline_02_moderate)
    t_02_vs_01, p_02_vs_01 = perform_significance_test(baseline_02_moderate, baseline_01_moderate)
    
    # Generate markdown report
    report = []
    report.append("# 📊 Validation Experiment Results\n")
    report.append(f"**Date:** {Path.cwd()}\n")
    report.append(f"**Total Runs:** {len(baseline_01_moderate) + len(baseline_02_moderate) + len(baseline_03_moderate)}\n\n")
    
    report.append("## 🎯 Primary Results: Car 3D Detection AP@0.7 (Moderate)\n\n")
    
    # Main comparison table
    table_data = [
        [
            "Baseline_01\n(Single-Scale HardVFE)",
            f"{stats_01['mean']:.2f}%",
            f"±{stats_01['std']:.2f}%",
            f"[{stats_01['ci_low']:.2f}, {stats_01['ci_high']:.2f}]",
            stats_01['n'],
            "-",
            "-"
        ],
        [
            "Baseline_02\n(Fixed Multi-Scale)",
            f"{stats_02['mean']:.2f}%",
            f"±{stats_02['std']:.2f}%",
            f"[{stats_02['ci_low']:.2f}, {stats_02['ci_high']:.2f}]",
            stats_02['n'],
            f"{stats_02['mean'] - stats_01['mean']:+.2f}%",
            f"p={p_02_vs_01:.4f}" if p_02_vs_01 < 0.10 else "n.s."
        ],
        [
            "**Baseline_03**\n**(Adaptive Learnable)**",
            f"**{stats_03['mean']:.2f}%**",
            f"**±{stats_03['std']:.2f}%**",
            f"**[{stats_03['ci_low']:.2f}, {stats_03['ci_high']:.2f}]**",
            stats_03['n'],
            f"**{stats_03['mean'] - stats_01['mean']:+.2f}%**",
            f"**p={p_03_vs_01:.4f}**" if p_03_vs_01 < 0.10 else "n.s."
        ]
    ]
    
    headers = ["Method", "Mean AP", "Std Dev", "95% CI", "N", "vs Baseline_01", "Significance"]
    report.append(tabulate(table_data, headers=headers, tablefmt="github"))
    report.append("\n\n")
    
    # Determine if results are conclusive
    improvement_03_vs_01 = stats_03['mean'] - stats_01['mean']
    is_significant = p_03_vs_01 < 0.05
    is_clear_winner = improvement_03_vs_01 > 2.0 and is_significant
    
    report.append("## 📈 Analysis\n\n")
    
    if is_clear_winner:
        report.append(f"### ✅ **VALIDATION SUCCESSFUL**\n\n")
        report.append(f"**Learnable multi-scale (Baseline_03) clearly outperforms both baselines:**\n\n")
        report.append(f"- **vs Single-Scale:** +{improvement_03_vs_01:.2f}% (p={p_03_vs_01:.4f}) {'✅ SIGNIFICANT' if p_03_vs_01 < 0.05 else ''}\n")
        report.append(f"- **vs Fixed Multi-Scale:** +{stats_03['mean'] - stats_02['mean']:.2f}% (p={p_03_vs_02:.4f}) {'✅ SIGNIFICANT' if p_03_vs_02 < 0.05 else ''}\n\n")
        report.append(f"**Conclusion:** Proceed with full multi-class evaluation and paper improvements.\n\n")
    elif improvement_03_vs_01 > 1.0:
        report.append(f"### ⚠️ **MODERATE IMPROVEMENT**\n\n")
        report.append(f"Learnable multi-scale shows improvement (+{improvement_03_vs_01:.2f}%), but:\n\n")
        if not is_significant:
            report.append(f"- ❌ Not statistically significant (p={p_03_vs_01:.4f} > 0.05)\n")
        if stats_03['std'] > 2.0:
            report.append(f"- ⚠️ High variance (±{stats_03['std']:.2f}%) indicates training instability\n")
        report.append(f"\n**Recommendation:** Investigate stability issues before expanding evaluation.\n\n")
    else:
        report.append(f"### ❌ **VALIDATION FAILED**\n\n")
        report.append(f"Learnable multi-scale does NOT clearly outperform baseline:\n\n")
        report.append(f"- Improvement: only {improvement_03_vs_01:.2f}%\n")
        report.append(f"- Statistical significance: p={p_03_vs_01:.4f}\n\n")
        report.append(f"**Recommendation:** Revisit method design before proceeding with paper.\n\n")
    
    # Detailed results by difficulty
    report.append("## 📋 Detailed Results: All Difficulty Levels\n\n")
    
    for difficulty in ['easy', 'moderate', 'hard']:
        report.append(f"### Car 3D AP@0.7 ({difficulty.capitalize()})\n\n")
        
        vals_01 = [r[f'car_3d_{difficulty}'] for r in results['baseline_01_single_scale'] if f'car_3d_{difficulty}' in r]
        vals_02 = [r[f'car_3d_{difficulty}'] for r in results['baseline_02_fixed_multiscale'] if f'car_3d_{difficulty}' in r]
        vals_03 = [r[f'car_3d_{difficulty}'] for r in results['baseline_03_adaptive'] if f'car_3d_{difficulty}' in r]
        
        stats_01_diff = compute_statistics(vals_01)
        stats_02_diff = compute_statistics(vals_02)
        stats_03_diff = compute_statistics(vals_03)
        
        table_data = [
            ["Baseline_01", f"{stats_01_diff['mean']:.2f}%", f"±{stats_01_diff['std']:.2f}%"],
            ["Baseline_02", f"{stats_02_diff['mean']:.2f}%", f"±{stats_02_diff['std']:.2f}%"],
            ["**Baseline_03**", f"**{stats_03_diff['mean']:.2f}%**", f"**±{stats_03_diff['std']:.2f}%**"]
        ]
        
        report.append(tabulate(table_data, headers=["Method", "Mean", "Std"], tablefmt="github"))
        report.append("\n\n")
    
    # Raw data
    report.append("## 📊 Raw Data\n\n")
    report.append("### Baseline_01 (Single-Scale)\n")
    report.append(f"Values: {[f'{v:.2f}' for v in baseline_01_moderate]}\n\n")
    
    report.append("### Baseline_02 (Fixed Multi-Scale)\n")
    report.append(f"Values: {[f'{v:.2f}' for v in baseline_02_moderate]}\n\n")
    
    report.append("### Baseline_03 (Adaptive Learnable)\n")
    report.append(f"Values: {[f'{v:.2f}' for v in baseline_03_moderate]}\n\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ Report generated: {output_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline_01 (Single-Scale):    {stats_01['mean']:.2f}% ± {stats_01['std']:.2f}%")
    print(f"Baseline_02 (Fixed Multi):     {stats_02['mean']:.2f}% ± {stats_02['std']:.2f}%")
    print(f"Baseline_03 (Adaptive):        {stats_03['mean']:.2f}% ± {stats_03['std']:.2f}%")
    print(f"\nImprovement (03 vs 01):        {improvement_03_vs_01:+.2f}%")
    print(f"Statistical Significance:      p={p_03_vs_01:.4f} {'✅' if p_03_vs_01 < 0.05 else '❌'}")
    print(f"\n{'✅ VALIDATION SUCCESSFUL' if is_clear_winner else '⚠️ NEEDS INVESTIGATION'}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze validation experiment results')
    parser.add_argument('--work-dir', type=str, default='work_dirs/validation_experiment',
                       help='Base work directory containing all runs')
    parser.add_argument('--output', type=str, default='VALIDATION_RESULTS.md',
                       help='Output markdown file')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    output_file = Path(args.output)
    
    if not work_dir.exists():
        print(f"❌ Error: Work directory not found: {work_dir}")
        return 1
    
    print(f"Collecting results from: {work_dir}")
    results = collect_results(work_dir)
    
    # Check if we have results
    total_runs = sum(len(v) for v in results.values())
    if total_runs == 0:
        print("❌ Error: No results found. Have the training runs completed?")
        return 1
    
    print(f"Found {total_runs} completed runs:")
    for baseline, runs in results.items():
        print(f"  - {baseline}: {len(runs)} runs")
    
    print("\nGenerating report...")
    generate_report(results, output_file)
    
    return 0


if __name__ == '__main__':
    exit(main())
