#!/usr/bin/env python3
"""
Test script to verify actual memory reduction with the optimized VFE.
"""

import torch
import torch.nn as nn
from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import (
    ImportanceGuidedMultiScaleVFE,
    MemoryOptimizedImportanceGuidedMultiScaleVFE
)
import tracemalloc
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_usage():
    """Test and compare memory usage between standard and optimized VFE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create test point cloud (similar to KITTI size)
    batch_size = 1
    num_points = 50000  # Typical KITTI point cloud size
    points = torch.randn(num_points, 4, device=device)  # x, y, z, intensity
    
    print(f"📊 Test setup: {num_points} points, batch size {batch_size}")
    print("=" * 60)
    
    # Test 1: Standard VFE
    print("🔧 Testing Standard ImportanceGuidedMultiScaleVFE...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    tracemalloc.start()
    mem_before_standard = get_memory_usage()
    
    standard_vfe = ImportanceGuidedMultiScaleVFE(
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        vfe_channels=[32, 64],
        fusion_channels=128,
        output_channels=64
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output_std, coors_std = standard_vfe(points)
    
    mem_after_standard = get_memory_usage()
    standard_memory = mem_after_standard - mem_before_standard
    
    if torch.cuda.is_available():
        cuda_memory_std = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.empty_cache()
    else:
        cuda_memory_std = 0
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   RAM usage: {standard_memory:.1f} MB")
    print(f"   CUDA memory: {cuda_memory_std:.1f} MB")
    print(f"   Output shape: {output_std.shape}")
    print(f"   Parameters: {sum(p.numel() for p in standard_vfe.parameters()):,}")
    
    # Clean up
    del standard_vfe, output_std, coors_std
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    
    # Test 2: Memory-Optimized VFE (Level 1)
    print("🚀 Testing MemoryOptimized VFE (Level 1)...")
    
    tracemalloc.start()
    mem_before_opt1 = get_memory_usage()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    opt_vfe_l1 = MemoryOptimizedImportanceGuidedMultiScaleVFE(
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        memory_optimization_level=1,
        importance_threshold=0.15,
        max_points_ratio=0.7,
        output_channels=64
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output_opt1, coors_opt1 = opt_vfe_l1(points)
    
    mem_after_opt1 = get_memory_usage()
    opt1_memory = mem_after_opt1 - mem_before_opt1
    
    if torch.cuda.is_available():
        cuda_memory_opt1 = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.empty_cache()
    else:
        cuda_memory_opt1 = 0
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   RAM usage: {opt1_memory:.1f} MB")
    print(f"   CUDA memory: {cuda_memory_opt1:.1f} MB")
    print(f"   Output shape: {output_opt1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in opt_vfe_l1.parameters()):,}")
    
    # Calculate savings
    ram_savings_l1 = (standard_memory - opt1_memory) / standard_memory * 100 if standard_memory > 0 else 0
    cuda_savings_l1 = (cuda_memory_std - cuda_memory_opt1) / cuda_memory_std * 100 if cuda_memory_std > 0 else 0
    
    print(f"   💾 RAM savings: {ram_savings_l1:.1f}%")
    print(f"   💾 CUDA savings: {cuda_savings_l1:.1f}%")
    
    # Clean up
    del opt_vfe_l1, output_opt1, coors_opt1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    
    # Test 3: Memory-Optimized VFE (Level 2 - EXTREME)
    print("🔥 Testing MemoryOptimized VFE (Level 2 - EXTREME)...")
    
    tracemalloc.start()
    mem_before_opt2 = get_memory_usage()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    opt_vfe_l2 = MemoryOptimizedImportanceGuidedMultiScaleVFE(
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        memory_optimization_level=2,  # EXTREME mode
        importance_threshold=0.25,
        max_points_ratio=0.5,  # Keep only 50% of points
        output_channels=64
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output_opt2, coors_opt2 = opt_vfe_l2(points)
    
    mem_after_opt2 = get_memory_usage()
    opt2_memory = mem_after_opt2 - mem_before_opt2
    
    if torch.cuda.is_available():
        cuda_memory_opt2 = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        cuda_memory_opt2 = 0
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   RAM usage: {opt2_memory:.1f} MB")
    print(f"   CUDA memory: {cuda_memory_opt2:.1f} MB")
    print(f"   Output shape: {output_opt2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in opt_vfe_l2.parameters()):,}")
    
    # Calculate savings
    ram_savings_l2 = (standard_memory - opt2_memory) / standard_memory * 100 if standard_memory > 0 else 0
    cuda_savings_l2 = (cuda_memory_std - cuda_memory_opt2) / cuda_memory_std * 100 if cuda_memory_std > 0 else 0
    
    print(f"   💾 RAM savings: {ram_savings_l2:.1f}%")
    print(f"   💾 CUDA savings: {cuda_savings_l2:.1f}%")
    
    # Print memory stats
    print(f"\n📈 Memory optimization stats:")
    opt_vfe_l2.print_memory_summary()
    
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON:")
    print(f"Standard VFE:     {standard_memory:.1f} MB RAM, {cuda_memory_std:.1f} MB CUDA")
    print(f"Optimized L1:     {opt1_memory:.1f} MB RAM, {cuda_memory_opt1:.1f} MB CUDA ({ram_savings_l1:.1f}% / {cuda_savings_l1:.1f}% saved)")
    print(f"Optimized L2:     {opt2_memory:.1f} MB RAM, {cuda_memory_opt2:.1f} MB CUDA ({ram_savings_l2:.1f}% / {cuda_savings_l2:.1f}% saved)")
    
    # Target check
    target_savings = 25.0  # 25% reduction target
    if cuda_savings_l2 >= target_savings:
        print(f"✅ SUCCESS: Level 2 achieves {cuda_savings_l2:.1f}% CUDA memory reduction (target: {target_savings}%)")
    else:
        print(f"⚠️ PARTIAL: Level 2 achieves {cuda_savings_l2:.1f}% CUDA memory reduction (target: {target_savings}%)")
    
    return {
        'standard': {'ram': standard_memory, 'cuda': cuda_memory_std},
        'opt_l1': {'ram': opt1_memory, 'cuda': cuda_memory_opt1, 'savings': (ram_savings_l1, cuda_savings_l1)},
        'opt_l2': {'ram': opt2_memory, 'cuda': cuda_memory_opt2, 'savings': (ram_savings_l2, cuda_savings_l2)}
    }

if __name__ == '__main__':
    print("🧪 Memory Reduction Test for Adaptive Voxelization VFE")
    print("=" * 60)
    
    try:
        results = test_memory_usage()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
