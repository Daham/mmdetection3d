#!/usr/bin/env python3
"""
Comprehensive test for different middle encoders with adaptive voxelization.

This script tests and compares:
1. SparseEncoder (vanilla SECOND)
2. AdaptiveSparseEncoderV3 (our full adaptive version)
3. AdaptiveSparseEncoderV3Simple (simplified adaptive version)
4. SparseUNet (U-Net style encoder)

Tests performance, memory usage, and adaptive capability.
"""

import sys
import os
sys.path.insert(0, '/Users/dahamp/Documents/academic/phd-repos/mmdetection3d')

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any
import traceback

# Import MMDetection3D modules
try:
    from mmdet3d.models.middle_encoders import (
        SparseEncoder, 
        AdaptiveSparseEncoderV3,
        AdaptiveSparseEncoderV3Simple,
        SparseUNet
    )
    from mmdet3d.models.voxel_encoders import AdaptiveVFE
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct environment and MMDetection3D is installed")
    sys.exit(1)

class MiddleEncoderTester:
    """Test suite for middle encoders with adaptive voxelization."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sparse_shape = [41, 1600, 1408]  # Standard KITTI shape
        self.batch_size = 2
        
        # Test configurations
        self.configs = {
            'SparseEncoder': {
                'type': SparseEncoder,
                'kwargs': {
                    'in_channels': 4,
                    'sparse_shape': self.sparse_shape,
                    'base_channels': 16,
                    'output_channels': 128
                }
            },
            'AdaptiveSparseEncoderV3': {
                'type': AdaptiveSparseEncoderV3,
                'kwargs': {
                    'in_channels': 4,
                    'sparse_shape': self.sparse_shape,
                    'base_channels': 16,
                    'output_channels': 128,
                    'adaptive_processing': True,
                    'adaptive_attention': True,
                    'multi_scale_fusion': True
                }
            },
            'AdaptiveSparseEncoderV3Simple': {
                'type': AdaptiveSparseEncoderV3Simple,
                'kwargs': {
                    'in_channels': 4,
                    'sparse_shape': self.sparse_shape,
                    'base_channels': 16,
                    'output_channels': 128,
                    'adaptive_channel_boost': 64
                }
            },
            'SparseUNet': {
                'type': SparseUNet,
                'kwargs': {
                    'in_channels': 4,
                    'sparse_shape': self.sparse_shape,
                    'base_channels': 16,
                    'output_channels': 128
                }
            }
        }
        
        self.results = {}
    
    def generate_test_data(self, num_voxels=1000):
        """Generate test data for middle encoder testing."""
        
        # Generate random voxel features
        voxel_features = torch.randn(num_voxels, 4, device=self.device)
        
        # Generate coordinates (batch_idx, z, y, x)
        coords = torch.zeros(num_voxels, 4, device=self.device, dtype=torch.int32)
        coords[:, 0] = torch.randint(0, self.batch_size, (num_voxels,))  # batch_idx
        coords[:, 1] = torch.randint(0, self.sparse_shape[0], (num_voxels,))  # z
        coords[:, 2] = torch.randint(0, self.sparse_shape[1], (num_voxels,))  # y  
        coords[:, 3] = torch.randint(0, self.sparse_shape[2], (num_voxels,))  # x
        
        # Generate adaptive information
        adaptive_info = {
            'adaptive_sizes': torch.tensor([0.8, 1.2, 1.0], device=self.device),  # Global adaptive sizes
            'density_info': torch.randn(num_voxels, device=self.device),
        }
        
        return voxel_features, coords, adaptive_info
    
    def test_encoder(self, name: str, config: Dict[str, Any], test_adaptive: bool = False):
        """Test a specific middle encoder."""
        
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = config['type'](**config['kwargs']).to(self.device)
            model.eval()
            
            # Generate test data
            voxel_features, coords, adaptive_info = self.generate_test_data()
            
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Input shape: voxel_features={voxel_features.shape}, coords={coords.shape}")
            
            # Test forward pass
            with torch.no_grad():
                start_time = time.time()
                
                # Test with and without adaptive info
                if test_adaptive and hasattr(model, 'forward') and 'adaptive_info' in model.forward.__code__.co_varnames:
                    output = model(voxel_features, coords, self.batch_size, adaptive_info)
                    print("✓ Adaptive forward pass successful")
                else:
                    output = model(voxel_features, coords, self.batch_size)
                    print("✓ Standard forward pass successful")
                
                forward_time = time.time() - start_time
            
            # Analyze output
            if isinstance(output, tuple):
                spatial_features = output[0]
                print(f"Output: tuple with spatial_features={spatial_features.shape}")
                if len(output) > 1:
                    print(f"Additional outputs: {len(output)-1} items")
            else:
                spatial_features = output
                print(f"Output shape: {spatial_features.shape}")
            
            # Memory usage
            if self.device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                print(f"Peak GPU memory: {memory_used:.1f} MB")
                torch.cuda.reset_peak_memory_stats()
            
            # Performance metrics
            self.results[name] = {
                'forward_time': forward_time,
                'output_shape': spatial_features.shape,
                'parameters': sum(p.numel() for p in model.parameters()),
                'adaptive_compatible': test_adaptive and 'adaptive_info' in model.forward.__code__.co_varnames,
                'success': True
            }
            
            print(f"Forward time: {forward_time:.4f}s")
            print(f"✓ {name} test completed successfully")
            
        except Exception as e:
            print(f"✗ {name} test failed: {str(e)}")
            traceback.print_exc()
            self.results[name] = {
                'success': False,
                'error': str(e)
            }
    
    def test_adaptive_capability(self):
        """Test adaptive capability specifically."""
        
        print(f"\n{'='*60}")
        print("Testing Adaptive Capability")
        print(f"{'='*60}")
        
        # Test AdaptiveVFE + AdaptiveSparseEncoderV3 combination
        try:
            # Create adaptive VFE
            vfe_config = {
                'max_num_points': 64,
                'voxel_size': [0.05, 0.05, 0.1],
                'point_cloud_range': [0, -40, -3, 70.4, 40, 1],
                'adaptive_type': 'density_based',
                'base_voxel_size': [0.05, 0.05, 0.1],
                'size_bounds': [0.5, 2.0],
                'learning_rate': 0.01
            }
            
            adaptive_vfe = AdaptiveVFE(**vfe_config).to(self.device)
            
            # Create adaptive middle encoder
            encoder_config = self.configs['AdaptiveSparseEncoderV3Simple']
            adaptive_encoder = encoder_config['type'](**encoder_config['kwargs']).to(self.device)
            
            # Generate point cloud data
            num_points = 2000
            points = torch.randn(num_points, 4, device=self.device)  # [x, y, z, intensity]
            
            # Test full pipeline
            with torch.no_grad():
                # VFE processing
                vfe_output = adaptive_vfe.forward(points.unsqueeze(0))  # Add batch dimension
                
                if isinstance(vfe_output, tuple):
                    voxel_features, coords = vfe_output[:2]
                    adaptive_info = vfe_output[2] if len(vfe_output) > 2 else None
                else:
                    voxel_features, coords = vfe_output, None
                    adaptive_info = None
                
                print(f"VFE output: voxel_features={voxel_features.shape}")
                
                if coords is not None and adaptive_info is not None:
                    # Middle encoder processing with adaptive info
                    encoder_output = adaptive_encoder(voxel_features, coords, 1, adaptive_info)
                    print(f"Adaptive encoder output: {encoder_output.shape}")
                    print("✓ Full adaptive pipeline successful")
                else:
                    print("⚠ Adaptive info not available, falling back to standard processing")
                    
        except Exception as e:
            print(f"✗ Adaptive capability test failed: {str(e)}")
            traceback.print_exc()
    
    def run_all_tests(self):
        """Run all encoder tests."""
        
        print("Middle Encoder Comparison Test")
        print(f"Device: {self.device}")
        print(f"Sparse shape: {self.sparse_shape}")
        print(f"Batch size: {self.batch_size}")
        
        # Test each encoder
        for name, config in self.configs.items():
            test_adaptive = 'Adaptive' in name
            self.test_encoder(name, config, test_adaptive)
        
        # Test full adaptive capability
        self.test_adaptive_capability()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = [name for name, result in self.results.items() if result.get('success', False)]
        failed_tests = [name for name, result in self.results.items() if not result.get('success', False)]
        
        print(f"Successful tests: {len(successful_tests)}")
        print(f"Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            print(f"\n✓ Successful encoders:")
            for name in successful_tests:
                result = self.results[name]
                print(f"  {name:30} | {result['parameters']:8,} params | {result['forward_time']:.4f}s | Adaptive: {result['adaptive_compatible']}")
        
        if failed_tests:
            print(f"\n✗ Failed encoders:")
            for name in failed_tests:
                result = self.results[name]
                print(f"  {name:30} | Error: {result['error']}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if successful_tests:
            # Find best adaptive encoder
            adaptive_encoders = [name for name in successful_tests if 'Adaptive' in name and self.results[name]['adaptive_compatible']]
            
            if adaptive_encoders:
                # Choose based on performance and complexity
                if 'AdaptiveSparseEncoderV3Simple' in adaptive_encoders:
                    print("🏆 RECOMMENDED: AdaptiveSparseEncoderV3Simple")
                    print("   - Good balance of adaptivity and performance")
                    print("   - Simpler implementation, easier to debug")
                    print("   - Compatible with existing SECOND pipeline")
                elif 'AdaptiveSparseEncoderV3' in adaptive_encoders:
                    print("🏆 RECOMMENDED: AdaptiveSparseEncoderV3")
                    print("   - Full adaptive capabilities")
                    print("   - Advanced attention and multi-scale processing")
                    print("   - Best for research and experimentation")
                
                print("\n📝 Next steps:")
                print("   1. Integrate chosen encoder into SECOND config")
                print("   2. Test with full training pipeline")
                print("   3. Compare detection performance vs vanilla SECOND")
                print("   4. Tune adaptive parameters for your dataset")
            else:
                print("⚠ No adaptive encoders working properly")
                print("   Fallback to standard SparseEncoder with manual optimization")
        else:
            print("❌ All tests failed - check environment and dependencies")


def main():
    """Run the middle encoder testing suite."""
    
    print("Middle Encoder Testing Suite")
    print("Testing different encoders for adaptive voxelization compatibility")
    
    tester = MiddleEncoderTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
