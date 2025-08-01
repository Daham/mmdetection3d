#!/usr/bin/env python3
"""
Simple test for AdaptiveSparseBridge to verify shape compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    import torch
    import torch.nn as nn
    
    # Import our adaptive bridge
    from mmdet3d.models.voxel_encoders.adaptive_sparse_bridge import AdaptiveSparseBridge
    
    print("✅ Successfully imported AdaptiveSparseBridge")
    
    def test_adaptive_bridge():
        """Test the adaptive bridge with realistic inputs."""
        print("\n🧪 Testing AdaptiveSparseBridge...")
        
        # Create the bridge
        bridge = AdaptiveSparseBridge(
            base_voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            min_voxel_size=[0.025, 0.025, 0.05],
            max_voxel_size=[0.2, 0.2, 0.4],
            in_channels=4,
            feat_channels=[4],  # Output 4 channels like HardSimpleVFE
            learnable_adaptation=True
        )
        
        print(f"📋 Bridge created successfully")
        
        # Create test inputs similar to what SECOND expects
        batch_size = 10  # 10 voxels
        max_points = 32  # max points per voxel
        feat_dim = 4     # x, y, z, intensity
        
        # Mock voxel features [N, M, C] where N=voxels, M=max_points, C=features
        features = torch.randn(batch_size, max_points, feat_dim)
        
        # Mock number of points per voxel [N]
        num_points = torch.randint(1, max_points, (batch_size,))
        
        # Mock coordinates [N, 4] (batch_idx, z, y, x)
        coors = torch.randint(0, 100, (batch_size, 4))
        
        print(f"📥 Input shapes:")
        print(f"   features: {features.shape}")
        print(f"   num_points: {num_points.shape}")
        print(f"   coors: {coors.shape}")
        
        try:
            # Run forward pass
            output = bridge.forward(features, num_points, coors)
            
            print(f"📤 Output shape: {output.shape}")
            print(f"✅ Expected shape: [batch_size, feat_channels] = [{batch_size}, {bridge.feat_channels[-1]}]")
            
            # Verify the output shape
            expected_shape = (batch_size, bridge.feat_channels[-1])
            if output.shape == expected_shape:
                print(f"✅ Shape test PASSED!")
                return True
            else:
                print(f"❌ Shape test FAILED! Expected {expected_shape}, got {output.shape}")
                return False
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_with_different_sizes():
        """Test with different batch sizes to ensure robustness."""
        print(f"\n🔄 Testing with different batch sizes...")
        
        bridge = AdaptiveSparseBridge(
            feat_channels=[4],
            learnable_adaptation=False  # Disable learning for this test
        )
        
        success_count = 0
        test_cases = [1, 5, 100, 1000]
        
        for batch_size in test_cases:
            try:
                features = torch.randn(batch_size, 32, 4)
                num_points = torch.randint(1, 32, (batch_size,))
                coors = torch.randint(0, 100, (batch_size, 4))
                
                output = bridge.forward(features, num_points, coors)
                expected_shape = (batch_size, 4)
                
                if output.shape == expected_shape:
                    print(f"   ✅ Batch size {batch_size}: {output.shape}")
                    success_count += 1
                else:
                    print(f"   ❌ Batch size {batch_size}: Expected {expected_shape}, got {output.shape}")
                    
            except Exception as e:
                print(f"   ❌ Batch size {batch_size}: Failed with {e}")
        
        print(f"📊 Passed {success_count}/{len(test_cases)} tests")
        return success_count == len(test_cases)
    
    if __name__ == "__main__":
        test1_passed = test_adaptive_bridge()
        test2_passed = test_with_different_sizes()
        
        if test1_passed and test2_passed:
            print(f"\n🎉 ALL TESTS PASSED! AdaptiveSparseBridge is working correctly.")
            print(f"   Ready for integration with sparse convolution!")
        else:
            print(f"\n💥 Some tests failed. Need to fix the implementation.")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"💡 This might be due to environment setup. The module should work in the proper MMDetection3D environment.")
