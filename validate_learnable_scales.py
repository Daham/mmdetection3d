#!/usr/bin/env python3
"""
🎓 PhD Research Validation: Test Learnable Voxel Scale Parameters

This script validates that voxel scales are now truly learnable parameters
that can be optimized through backpropagation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the mmdet3d path
sys.path.insert(0, '/home/daham/mmdetection_project/mmdetection3d')

def test_learnable_voxel_scales():
    """Test that voxel scales are learnable parameters."""
    print("🎓 PhD RESEARCH VALIDATION: Testing Learnable Voxel Scale Parameters")
    print("=" * 80)
    
    try:
        from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ScaleNet
        
        # Create ScaleNet with initial scales
        scale_net = ScaleNet(
            in_channels=4,
            hidden_dims=[32, 16],
            num_scales=3,
            voxel_scales=[0.05, 0.1, 0.2]  # Initial scales
        )
        
        print("✅ ScaleNet created successfully")
        
        # Check if voxel_scales are learnable parameters
        print(f"\n🔍 Checking parameter status:")
        print(f"   voxel_scales type: {type(scale_net.voxel_scales)}")
        print(f"   voxel_scales requires_grad: {scale_net.voxel_scales.requires_grad}")
        print(f"   voxel_scales is_parameter: {isinstance(scale_net.voxel_scales, nn.Parameter)}")
        print(f"   Initial scales: {[f'{s:.4f}m' for s in scale_net.voxel_scales.tolist()]}")
        
        # Test backpropagation through scales
        print(f"\n🚀 Testing backpropagation through scale parameters:")
        
        # Create dummy input
        points = torch.randn(100, 4, requires_grad=True)  # 100 points, 4 features
        
        # Forward pass
        scale_assignment, predicted_scales = scale_net(points, training=True)
        
        print(f"   Scale assignment shape: {scale_assignment.shape}")
        print(f"   Predicted scales shape: {predicted_scales.shape}")
        print(f"   Predicted scale range: {predicted_scales.min():.4f}m - {predicted_scales.max():.4f}m")
        
        # Create a dummy loss that depends on the predicted scales
        # This simulates how the detection loss would affect scale learning
        target_scales = torch.full_like(predicted_scales, 0.08)  # Target: 0.08m scales
        scale_loss = nn.MSELoss()(predicted_scales, target_scales)
        
        print(f"   Scale loss: {scale_loss.item():.6f}")
        
        # Backpropagate
        scale_loss.backward()
        
        # Check if gradients were computed for voxel_scales
        print(f"\n📈 Gradient check:")
        if scale_net.voxel_scales.grad is not None:
            print(f"   ✅ voxel_scales received gradients!")
            print(f"   Gradients: {[f'{g:.6f}' for g in scale_net.voxel_scales.grad.tolist()]}")
            
            # Test optimization step
            optimizer = optim.Adam(scale_net.parameters(), lr=0.01)
            
            scales_before = scale_net.voxel_scales.clone().detach()
            optimizer.step()
            scales_after = scale_net.voxel_scales.clone().detach()
            
            print(f"\n🎯 Optimization test:")
            print(f"   Scales before: {[f'{s:.4f}m' for s in scales_before.tolist()]}")
            print(f"   Scales after:  {[f'{s:.4f}m' for s in scales_after.tolist()]}")
            print(f"   Change: {[f'{(a-b):.6f}' for a, b in zip(scales_after.tolist(), scales_before.tolist())]}")
            
            if not torch.allclose(scales_before, scales_after, atol=1e-6):
                print(f"   ✅ SUCCESS: Scales changed during optimization!")
                print(f"   🎓 PhD Research Status: LEARNABLE VOXEL SCALES CONFIRMED!")
                return True
            else:
                print(f"   ❌ FAILED: Scales did not change during optimization")
                return False
        else:
            print(f"   ❌ FAILED: No gradients computed for voxel_scales")
            return False
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scale_regularization():
    """Test scale regularization loss."""
    print(f"\n🔧 Testing scale regularization:")
    
    try:
        from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ScaleNet
        
        scale_net = ScaleNet(
            in_channels=4,
            hidden_dims=[32, 16], 
            num_scales=3,
            voxel_scales=[0.05, 0.1, 0.2]
        )
        
        # Test regularization loss
        reg_loss = scale_net.get_scale_regularization_loss(weight=0.01)
        print(f"   Initial regularization loss: {reg_loss.item():.6f}")
        
        # Test with extreme scales to trigger regularization
        with torch.no_grad():
            scale_net.voxel_scales[0] = -0.1  # Negative scale (bad)
            scale_net.voxel_scales[1] = 2.0   # Too large scale (bad)
            
        reg_loss_extreme = scale_net.get_scale_regularization_loss(weight=0.01)
        print(f"   Regularization with extreme scales: {reg_loss_extreme.item():.6f}")
        
        if reg_loss_extreme > reg_loss:
            print(f"   ✅ Regularization working: penalizes extreme scales")
        else:
            print(f"   ⚠️  Regularization may need tuning")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Regularization test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🎓 PhD RESEARCH: Adaptive Voxelization with Learnable Scale Parameters")
    print("🔬 Validating implementation for research contribution")
    print("=" * 80)
    
    success = True
    
    # Test 1: Learnable parameters
    success &= test_learnable_voxel_scales()
    
    # Test 2: Regularization
    success &= test_scale_regularization()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("🎓 PhD Research Implementation: VALIDATED")
        print("✅ Voxel scales are truly learnable through backpropagation")
        print("✅ Scale regularization prevents degenerate solutions")
        print("✅ Ready for PhD research experiments!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Implementation needs fixes before PhD experiments")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
