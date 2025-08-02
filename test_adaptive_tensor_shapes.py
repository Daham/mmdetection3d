#!/usr/bin/env python3
"""
Test script to validate our AdaptiveSparseEncoder tensor shape fix

This confirms that the encoder outputs the correct dense tensor format
for the backbone, solving the RuntimeError we encountered.
"""

import torch
import numpy as np
import sys
import os

# Add mmdet3d to path
sys.path.insert(0, '/Users/dahamp/Documents/academic/phd-repos/mmdetection3d')

from mmdet3d.models.voxel_encoders.adaptive_sparse_bridge import AdaptiveSparseBridge
from mmdet3d.models.middle_encoders.adaptive_sparse_encoder import AdaptiveSparseEncoder

def test_adaptive_pipeline():
    """Test the full adaptive voxelization pipeline with correct tensor shapes"""
    
    print("🧪 Testing Adaptive Voxelization Pipeline Tensor Shapes...")
    
    # Create test data
    batch_size = 2
    num_voxels = 1000
    num_points_per_voxel = 5
    point_dim = 4
    
    # Simulate voxel features and coordinates
    voxel_features = torch.randn(num_voxels, num_points_per_voxel, point_dim)
    voxel_coords = torch.randint(0, 100, (num_voxels, 4))  # [batch, z, y, x]
    voxel_coords[:, 0] = torch.randint(0, batch_size, (num_voxels,))  # Valid batch indices
    voxel_num_points = torch.randint(1, num_points_per_voxel + 1, (num_voxels,))
    
    print(f"✅ Input shapes:")
    print(f"   - voxel_features: {voxel_features.shape}")  
    print(f"   - voxel_coords: {voxel_coords.shape}")
    print(f"   - voxel_num_points: {voxel_num_points.shape}")
    
    # 1. Test AdaptiveSparseBridge (voxel encoder)
    print(f"\n1️⃣ Testing AdaptiveSparseBridge...")
    voxel_encoder = AdaptiveSparseBridge(
        num_features=point_dim,
        spatial_encoding_dim=32,
        voxel_predictor_hidden=64,
        voxel_aware_hidden=64,
        min_voxel_size=0.05,
        max_voxel_size=0.5
    )
    
    # Forward pass through voxel encoder
    with torch.no_grad():
        encoded_features = voxel_encoder(
            voxel_features, voxel_num_points, voxel_coords
        )
        # Get learned voxel sizes from encoder
        learned_voxel_sizes = voxel_encoder.last_voxel_sizes
    
    print(f"✅ AdaptiveSparseBridge outputs:")
    print(f"   - encoded_features: {encoded_features.shape}")
    print(f"   - learned_voxel_sizes: {learned_voxel_sizes.shape}")
    print(f"   - voxel size range: [{learned_voxel_sizes.min():.3f}, {learned_voxel_sizes.max():.3f}]")
    
    # 2. Test AdaptiveSparseEncoder (middle encoder)
    print(f"\n2️⃣ Testing AdaptiveSparseEncoder...")
    middle_encoder = AdaptiveSparseEncoder(
        in_channels=encoded_features.shape[1],
        sparse_shape=[41, 800, 704],  # Smaller for testing
        output_channels=128,
        num_size_groups=3,
        size_group_ranges=[(0.05, 0.20), (0.20, 0.35), (0.35, 0.50)]
    )
    
    # Forward pass through middle encoder
    with torch.no_grad():
        try:
            spatial_features = middle_encoder(
                voxel_features=encoded_features,
                coors=voxel_coords,
                batch_size=batch_size,
                voxel_sizes=learned_voxel_sizes
            )
            
            print(f"✅ AdaptiveSparseEncoder output:")
            print(f"   - spatial_features: {spatial_features.shape}")
            print(f"   - Expected format: [batch_size, channels, height, width]")
            
            # Validate tensor shape
            expected_dims = 4  # Should be 4D for backbone
            if len(spatial_features.shape) == expected_dims:
                print(f"   ✅ Correct 4D tensor shape for backbone!")
                print(f"   ✅ Batch size: {spatial_features.shape[0]}")
                print(f"   ✅ Channels: {spatial_features.shape[1]}")
                print(f"   ✅ Height: {spatial_features.shape[2]}")
                print(f"   ✅ Width: {spatial_features.shape[3]}")
                
                # Test that it's compatible with a simple 2D conv (like backbone)
                test_conv = torch.nn.Conv2d(spatial_features.shape[1], 64, 3, padding=1)
                with torch.no_grad():
                    conv_output = test_conv(spatial_features)
                    print(f"   ✅ Compatible with Conv2d: {conv_output.shape}")
                    
                return True
            else:
                print(f"   ❌ Wrong tensor shape! Got {len(spatial_features.shape)}D, expected {expected_dims}D")
                return False
                
        except Exception as e:
            print(f"   ❌ Error in AdaptiveSparseEncoder: {e}")
            return False

def main():
    """Run the tensor shape validation test"""
    print("🔬 Adaptive Voxelization Tensor Shape Validation")
    print("=" * 60)
    
    try:
        success = test_adaptive_pipeline()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 SUCCESS! Adaptive pipeline produces correct tensor shapes")
            print("✅ Fixed the RuntimeError: Expected 4D tensor for conv2d")
            print("✅ AdaptiveSparseEncoder now outputs dense 4D tensors")
            print("✅ Pipeline is ready for training with backbone!")
        else:
            print("❌ FAILED! Tensor shapes are still incorrect")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
