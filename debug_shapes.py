#!/usr/bin/env python3
"""
Debug script to understand the exact shape flow in adaptive voxelization.
This will help us fix the sparse convolution shape mismatch.
"""

import torch
import numpy as np
from mmdet3d.models.voxel_encoders.adaptive_sparse_bridge import AdaptiveSparseBridge

def debug_shapes():
    print("🔍 Debugging AdaptiveSparseBridge shapes...")
    
    # Create a simple instance
    bridge = AdaptiveSparseBridge(
        base_voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        min_voxel_size=[0.025, 0.025, 0.05],
        max_voxel_size=[0.2, 0.2, 0.4],
        in_channels=4,
        feat_channels=[4]
    )
    
    # Create mock input similar to what VFE receives
    batch_size = 2
    max_points = 32
    feat_dim = 4  # x, y, z, intensity
    
    # Mock voxel features [batch_size, max_points, feat_dim]
    features = torch.randn(batch_size, max_points, feat_dim)
    
    # Mock number of points per voxel
    num_points = torch.tensor([25, 30])  # Two voxels with 25 and 30 points
    
    # Mock coordinates (not used in current implementation)
    coors = torch.zeros(batch_size, 4, dtype=torch.long)
    
    print(f"Input shapes:")
    print(f"  features: {features.shape}")
    print(f"  num_points: {num_points.shape}")
    print(f"  coors: {coors.shape}")
    
    # Run forward pass
    try:
        output = bridge.forward(features, num_points, coors)
        print(f"✅ Output shape: {output.shape}")
        print(f"   Expected by SparseEncoder: [N, 4] where N is number of voxels")
        
        # Test what happens with more realistic sizes
        print(f"\n🧪 Testing with realistic voxel batch...")
        
        # Simulate a realistic batch (e.g., 1000 voxels)
        real_batch = 1000
        real_features = torch.randn(real_batch, max_points, feat_dim)
        real_num_points = torch.randint(1, max_points, (real_batch,))
        real_coors = torch.zeros(real_batch, 4, dtype=torch.long)
        
        real_output = bridge.forward(real_features, real_num_points, real_coors)
        print(f"✅ Realistic output shape: {real_output.shape}")
        
        return real_output
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_sparse_encoder_expectations():
    """Check what SparseEncoder expects."""
    print(f"\n🔍 Debugging SparseEncoder expectations...")
    
    try:
        from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
        
        encoder = SparseEncoder(
            in_channels=4,
            sparse_shape=[41, 1600, 1408],
            order=('conv', 'norm', 'act')
        )
        
        print(f"SparseEncoder created successfully")
        print(f"  in_channels: 4")
        print(f"  sparse_shape: [41, 1600, 1408]")
        
        # The SparseEncoder expects:
        # - voxel_features: [N, C] where N is number of voxels, C is channels (4)
        # - coordinates: [N, 4] where N is number of voxels, 4 is (batch_idx, z, y, x)
        
        print(f"\n📋 SparseEncoder input requirements:")
        print(f"  - voxel_features: [N, 4] tensor")
        print(f"  - coordinates: [N, 4] tensor (batch_idx, z, y, x)")
        
        return encoder
        
    except ImportError as e:
        print(f"❌ Could not import SparseEncoder: {e}")
        return None

if __name__ == "__main__":
    output = debug_shapes()
    encoder = debug_sparse_encoder_expectations()
    
    if output is not None and encoder is not None:
        print(f"\n🎯 Shape compatibility check:")
        print(f"  AdaptiveSparseBridge output: {output.shape}")
        print(f"  SparseEncoder expects: [N, 4]")
        
        if output.shape[1] == 4:
            print(f"  ✅ Compatible!")
        else:
            print(f"  ❌ Incompatible! Need to fix output channels.")
            
    print(f"\n💡 The issue is likely in how we construct the voxel data for SparseEncoder.")
    print(f"   SparseEncoder needs both voxel_features AND coordinates, not just features.")
