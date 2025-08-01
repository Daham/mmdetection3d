#!/usr/bin/env python3
"""
Comprehensive Test Script for Multi-Resolution Adaptive Voxelization

This script tests and validates the complete adaptive voxelization approach,
including both the Enhanced Adaptive VFE and Multi-Resolution Sparse Encoder.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe import EnhancedAdaptiveVFE
from mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder import MultiResolutionSparseEncoder


def create_test_point_cloud(num_points=1000, point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
    """Create a realistic test point cloud with varying densities."""
    
    # Create points in different density regions
    dense_region_points = int(num_points * 0.4)  # 40% in dense regions
    sparse_region_points = int(num_points * 0.3)  # 30% in sparse regions
    medium_region_points = num_points - dense_region_points - sparse_region_points
    
    points = []
    
    # Dense region (close to origin, high density)
    dense_x = np.random.uniform(10, 30, dense_region_points)
    dense_y = np.random.uniform(-10, 10, dense_region_points)
    dense_z = np.random.uniform(-1, 0, dense_region_points)
    dense_intensity = np.random.uniform(0.5, 1.0, dense_region_points)
    dense_points = np.column_stack([dense_x, dense_y, dense_z, dense_intensity])
    points.append(dense_points)
    
    # Sparse region (far from origin, low density)
    sparse_x = np.random.uniform(50, 70, sparse_region_points)
    sparse_y = np.random.uniform(-35, 35, sparse_region_points)
    sparse_z = np.random.uniform(-2, 1, sparse_region_points)
    sparse_intensity = np.random.uniform(0.1, 0.5, sparse_region_points)
    sparse_points = np.column_stack([sparse_x, sparse_y, sparse_z, sparse_intensity])
    points.append(sparse_points)
    
    # Medium density region
    medium_x = np.random.uniform(30, 50, medium_region_points)
    medium_y = np.random.uniform(-20, 20, medium_region_points)
    medium_z = np.random.uniform(-1.5, 0.5, medium_region_points)
    medium_intensity = np.random.uniform(0.3, 0.8, medium_region_points)
    medium_points = np.column_stack([medium_x, medium_y, medium_z, medium_intensity])
    points.append(medium_points)
    
    # Combine all points
    all_points = np.vstack(points)
    return torch.from_numpy(all_points).float()


def create_test_voxel_data(points, voxel_size=(0.05, 0.05, 0.1), max_points_per_voxel=5):
    """Create test voxel data from point cloud."""
    from mmdet3d.ops import DynamicScatter
    
    # Simple voxelization (mock implementation)
    batch_size = 1
    num_voxels = min(1000, len(points) // max_points_per_voxel)
    
    # Create mock voxel features
    voxel_features = torch.zeros(batch_size, num_voxels, max_points_per_voxel, 4)
    for i in range(num_voxels):
        start_idx = i * max_points_per_voxel
        end_idx = min(start_idx + max_points_per_voxel, len(points))
        actual_points = end_idx - start_idx
        if actual_points > 0:
            voxel_features[0, i, :actual_points] = points[start_idx:end_idx]
    
    # Create mock coordinates
    coordinates = torch.zeros(num_voxels, 4, dtype=torch.int32)  # [batch_idx, z, y, x]
    for i in range(num_voxels):
        coordinates[i, 0] = 0  # batch index
        coordinates[i, 1] = i % 41  # z
        coordinates[i, 2] = (i // 41) % 100  # y
        coordinates[i, 3] = i // (41 * 100)  # x
    
    # Number of points per voxel
    num_points = torch.ones(num_voxels, dtype=torch.int32) * max_points_per_voxel
    
    return voxel_features, coordinates, num_points


def test_enhanced_adaptive_vfe():
    """Test the Enhanced Adaptive VFE module."""
    print("=" * 60)
    print("Testing Enhanced Adaptive VFE")
    print("=" * 60)
    
    # Create test model
    model = EnhancedAdaptiveVFE(
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=True,
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='multi_scale',
        num_scales=3,
        provide_multi_res_info=True
    )
    
    print(f"Model created successfully: {type(model).__name__}")
    print(f"Adaptation method: {model.adaptation_method}")
    print(f"Number of scales: {model.num_scales}")
    
    # Create test data
    points = create_test_point_cloud(2000)
    print(f"Created point cloud with {len(points)} points")
    print(f"Point cloud range: x=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    voxel_features, coordinates, num_points = create_test_voxel_data(points)
    print(f"Voxel features shape: {voxel_features.shape}")
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"Num points shape: {num_points.shape}")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(voxel_features, num_points, coordinates)
        
        print(f"✓ Forward pass successful!")
        print(f"Output keys: {list(output.keys()) if isinstance(output, dict) else 'Not a dict'}")
        
        if isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"Output shape: {output.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_multi_resolution_sparse_encoder():
    """Test the Multi-Resolution Sparse Encoder module."""
    print("=" * 60)
    print("Testing Multi-Resolution Sparse Encoder")
    print("=" * 60)
    
    # Create test model
    model = MultiResolutionSparseEncoder(
        base_voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        resolution_levels=[0.5, 1.0, 2.0],
        in_channels=128,
        out_channels=256,
        assignment_threshold=0.1,
        fusion_method='attention'
    )
    
    print(f"Model created successfully: {type(model).__name__}")
    print(f"Resolution levels: {model.resolution_levels}")
    print(f"Fusion method: {model.fusion_method}")
    print(f"Voxel sizes per level:")
    for level, voxel_size in model.voxel_sizes.items():
        print(f"  {level}: {voxel_size}")
    
    # Create test sparse tensor
    batch_size = 1
    num_voxels = 500
    
    # Create mock coordinates for sparse tensor
    indices = torch.zeros(num_voxels, 4, dtype=torch.int32)
    indices[:, 0] = 0  # batch index
    indices[:, 1] = torch.randint(0, 41, (num_voxels,))  # z
    indices[:, 2] = torch.randint(0, 100, (num_voxels,))  # y
    indices[:, 3] = torch.randint(0, 100, (num_voxels,))  # x
    
    # Create mock features
    features = torch.randn(num_voxels, 128)
    
    # Create mock adaptive info
    adaptive_info = {
        'voxel_scales': torch.rand(num_voxels, 3) * 1.5 + 0.5,  # Scale factors [0.5, 2.0]
        'importance_scores': torch.rand(num_voxels),
        'density_info': torch.rand(num_voxels)
    }
    
    print(f"Test data created:")
    print(f"  Features shape: {features.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Adaptive info keys: {list(adaptive_info.keys())}")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(features, indices, [41, 100, 100], adaptive_info)
        
        print(f"✓ Forward pass successful!")
        print(f"Output type: {type(output)}")
        
        if hasattr(output, 'features') and hasattr(output, 'indices'):
            print(f"Output features shape: {output.features.shape}")
            print(f"Output indices shape: {output.indices.shape}")
        elif isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        else:
            print(f"Output shape: {output.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_full_pipeline():
    """Test the complete pipeline with both modules."""
    print("=" * 60)
    print("Testing Complete Multi-Resolution Adaptive Pipeline")
    print("=" * 60)
    
    # Create models
    vfe_model = EnhancedAdaptiveVFE(
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=True,
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='multi_scale',
        num_scales=3,
        provide_multi_res_info=True
    )
    
    sparse_encoder = MultiResolutionSparseEncoder(
        base_voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        resolution_levels=[0.5, 1.0, 2.0],
        in_channels=128,
        out_channels=256,
        assignment_threshold=0.1,
        fusion_method='attention'
    )
    
    print("Both models created successfully")
    
    # Create test data
    points = create_test_point_cloud(2000)
    voxel_features, coordinates, num_points = create_test_voxel_data(points)
    
    try:
        # Step 1: VFE processing
        vfe_model.eval()
        sparse_encoder.eval()
        
        with torch.no_grad():
            vfe_output = vfe_model(voxel_features, num_points, coordinates)
            print("✓ VFE processing successful")
            
            # Extract features and adaptive info
            if isinstance(vfe_output, dict):
                features = vfe_output.get('features', vfe_output.get('voxel_features'))
                adaptive_info = vfe_output.get('adaptive_info', {})
            else:
                features = vfe_output
                adaptive_info = {}
            
            print(f"VFE features shape: {features.shape}")
            print(f"Adaptive info keys: {list(adaptive_info.keys())}")
            
            # Step 2: Multi-resolution sparse encoding
            sparse_shape = [41, 100, 100]  # Simplified for testing
            encoder_output = sparse_encoder(features, coordinates, sparse_shape, adaptive_info)
            print("✓ Multi-resolution sparse encoding successful")
            
            if hasattr(encoder_output, 'features'):
                print(f"Final output features shape: {encoder_output.features.shape}")
                print(f"Final output indices shape: {encoder_output.indices.shape}")
            else:
                print(f"Final output shape: {encoder_output.shape}")
            
            print("✓ Complete pipeline test successful!")
            return True
            
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_performance_analysis():
    """Run performance analysis of the multi-resolution approach."""
    print("=" * 60)
    print("Performance Analysis")
    print("=" * 60)
    
    import time
    
    # Create models
    standard_vfe = EnhancedAdaptiveVFE(
        in_channels=4,
        feat_channels=[64, 128],
        adaptation_method='density',  # Standard single-scale
        num_scales=1
    )
    
    multi_res_vfe = EnhancedAdaptiveVFE(
        in_channels=4,
        feat_channels=[64, 128],
        adaptation_method='multi_scale',  # Multi-scale
        num_scales=3
    )
    
    # Test data
    points = create_test_point_cloud(5000)
    voxel_features, coordinates, num_points = create_test_voxel_data(points, max_points_per_voxel=10)
    
    # Benchmark standard approach
    standard_vfe.eval()
    times = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            _ = standard_vfe(voxel_features, num_points, coordinates)
        times.append(time.time() - start)
    
    standard_time = np.mean(times[2:])  # Skip first 2 for warmup
    print(f"Standard VFE average time: {standard_time:.4f}s")
    
    # Benchmark multi-resolution approach
    multi_res_vfe.eval()
    times = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            _ = multi_res_vfe(voxel_features, num_points, coordinates)
        times.append(time.time() - start)
    
    multi_res_time = np.mean(times[2:])  # Skip first 2 for warmup
    print(f"Multi-resolution VFE average time: {multi_res_time:.4f}s")
    print(f"Overhead: {(multi_res_time / standard_time - 1) * 100:.1f}%")


def main():
    """Run all tests."""
    print("Multi-Resolution Adaptive Voxelization Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        device = torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    tests = [
        ("Enhanced Adaptive VFE", test_enhanced_adaptive_vfe),
        ("Multi-Resolution Sparse Encoder", test_multi_resolution_sparse_encoder),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Performance analysis
    try:
        print(f"\n{'='*20} Performance Analysis {'='*20}")
        run_performance_analysis()
        results["Performance Analysis"] = True
    except Exception as e:
        print(f"Performance analysis failed: {e}")
        results["Performance Analysis"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The multi-resolution adaptive voxelization system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
