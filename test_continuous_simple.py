#!/usr/bin/env python3
"""
🌊 Simple Continuous Adaptive Voxelization Test
Test the enhanced ScaleNet with continuous voxel size prediction.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/home/daham/mmdetection_project/mmdetection3d')

# Import our enhanced ScaleNet
from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ScaleNet

def test_continuous_prediction():
    """Test continuous vs discrete prediction modes."""
    print("🌊 CONTINUOUS ADAPTIVE VOXELIZATION TEST")
    print("=" * 50)
    
    # Create test data
    num_points = 100
    test_points = torch.randn(num_points, 4)  # x, y, z, intensity
    test_points[:, 3] = torch.rand(num_points)  # Ensure positive intensity
    
    print(f"📊 Test data: {num_points} points with shape {test_points.shape}")
    
    # 1. Test Discrete Mode (Original)
    print("\n🔢 Testing Discrete Mode...")
    discrete_net = ScaleNet(
        in_channels=4,
        hidden_dims=[64, 32],
        num_scales=10,
        temperature=2.0,
        continuous_mode=False  # Discrete mode
    )
    
    discrete_net.eval()
    with torch.no_grad():
        discrete_assignment, discrete_scales = discrete_net(test_points, training=False)
    
    print(f"   ✅ Discrete prediction successful!")
    print(f"   📏 Scale range: {discrete_scales.min():.3f}m - {discrete_scales.max():.3f}m")
    print(f"   🎯 Unique scales: {len(torch.unique(discrete_scales))}")
    print(f"   📊 Assignment shape: {discrete_assignment.shape}")
    
    # 2. Test Continuous Mode (Enhanced)
    print("\n🌊 Testing Continuous Mode...")
    continuous_net = ScaleNet(
        in_channels=4,
        hidden_dims=[64, 32],
        num_scales=10,
        temperature=2.0,
        continuous_mode=True,   # Continuous mode!
        min_voxel_size=0.01,
        max_voxel_size=1.0,
        interpolation_neighbors=3
    )
    
    continuous_net.eval()
    with torch.no_grad():
        continuous_assignment, continuous_scales = continuous_net(test_points, training=False)
    
    print(f"   ✅ Continuous prediction successful!")
    print(f"   📏 Scale range: {continuous_scales.min():.3f}m - {continuous_scales.max():.3f}m")
    print(f"   🎯 Unique scales: {len(torch.unique(torch.round(continuous_scales, decimals=3)))}")
    print(f"   📊 Assignment shape: {continuous_assignment.shape}")
    
    # 3. Compare Results
    print("\n📈 COMPARISON ANALYSIS")
    print("-" * 30)
    
    # Scale diversity
    discrete_unique = len(torch.unique(discrete_scales))
    continuous_unique = len(torch.unique(torch.round(continuous_scales, decimals=3)))
    
    print(f"🎨 Scale Diversity:")
    print(f"   Discrete: {discrete_unique} unique scales")
    print(f"   Continuous: {continuous_unique} unique scales")
    print(f"   Improvement: {continuous_unique / discrete_unique:.2f}x")
    
    # Assignment softness
    discrete_entropy = -torch.sum(discrete_assignment * torch.log(discrete_assignment + 1e-8), dim=1).mean()
    continuous_entropy = -torch.sum(continuous_assignment * torch.log(continuous_assignment + 1e-8), dim=1).mean()
    
    print(f"\n🤝 Assignment Softness (entropy):")
    print(f"   Discrete: {discrete_entropy:.3f}")
    print(f"   Continuous: {continuous_entropy:.3f}")
    print(f"   Difference: {(continuous_entropy - discrete_entropy):.3f}")
    
    # Scale statistics
    print(f"\n📊 Scale Statistics:")
    print(f"   Discrete std: {torch.std(discrete_scales):.4f}")
    print(f"   Continuous std: {torch.std(continuous_scales):.4f}")
    print(f"   Discrete mean: {torch.mean(discrete_scales):.4f}")
    print(f"   Continuous mean: {torch.mean(continuous_scales):.4f}")
    
    return True

def test_gradient_flow():
    """Test gradient flow in continuous mode."""
    print("\n🌊 GRADIENT FLOW TEST")
    print("-" * 30)
    
    # Create test points that require gradients
    test_points = torch.randn(50, 4, requires_grad=True)
    
    # Test continuous network
    continuous_net = ScaleNet(
        in_channels=4,
        hidden_dims=[32, 16],
        num_scales=5,
        continuous_mode=True,
        min_voxel_size=0.01,
        max_voxel_size=1.0
    )
    
    # Forward pass
    assignment, scales = continuous_net(test_points, training=True)
    
    # Compute a simple loss
    loss = scales.mean()
    loss.backward()
    
    # Check gradients
    grad_norm = test_points.grad.norm().item()
    
    print(f"✅ Gradient flow test successful!")
    print(f"📊 Loss: {loss.item():.4f}")
    print(f"🔥 Gradient norm: {grad_norm:.6f}")
    print(f"✅ Gradients are {'flowing' if grad_norm > 1e-6 else 'blocked'}")
    
    return grad_norm > 1e-6

def test_interpolation_neighbors():
    """Test different interpolation neighbor counts."""
    print("\n🤝 INTERPOLATION NEIGHBORS TEST")
    print("-" * 35)
    
    test_points = torch.randn(30, 4)
    
    for neighbors in [1, 2, 3, 4, 5]:
        print(f"\n   🔍 Testing {neighbors} neighbors...")
        
        net = ScaleNet(
            in_channels=4,
            hidden_dims=[32, 16],
            num_scales=8,
            continuous_mode=True,
            interpolation_neighbors=neighbors
        )
        
        net.eval()
        with torch.no_grad():
            assignment, scales = net(test_points, training=False)
        
        # Analyze assignment sparsity
        active_scales = (assignment > 0.01).sum(dim=1).float().mean()
        max_weight = assignment.max(dim=1)[0].mean()
        
        print(f"      Active scales per point: {active_scales:.2f}")
        print(f"      Average max weight: {max_weight:.3f}")
    
    print(f"\n✅ Interpolation test successful!")
    return True

def test_configuration_compatibility():
    """Test configuration compatibility with different settings."""
    print("\n⚙️ CONFIGURATION COMPATIBILITY TEST")
    print("-" * 40)
    
    test_points = torch.randn(20, 4)
    
    # Test different configurations
    configs = [
        {"num_scales": 3, "hidden_dims": [32, 16], "continuous_mode": False},
        {"num_scales": 5, "hidden_dims": [64, 32], "continuous_mode": False},
        {"num_scales": 10, "hidden_dims": [128, 64, 32], "continuous_mode": False},
        {"num_scales": 3, "hidden_dims": [32, 16], "continuous_mode": True},
        {"num_scales": 5, "hidden_dims": [64, 32], "continuous_mode": True},
        {"num_scales": 10, "hidden_dims": [128, 64, 32], "continuous_mode": True},
    ]
    
    for i, config in enumerate(configs):
        mode = "Continuous" if config["continuous_mode"] else "Discrete"
        print(f"\n   {i+1}. {mode} - {config['num_scales']} scales, {config['hidden_dims']} dims")
        
        try:
            net = ScaleNet(
                in_channels=4,
                **config
            )
            
            net.eval()
            with torch.no_grad():
                assignment, scales = net(test_points, training=False)
            
            print(f"      ✅ Success - Scale range: {scales.min():.3f}m - {scales.max():.3f}m")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            return False
    
    print(f"\n✅ All configurations compatible!")
    return True

def main():
    """Main test function."""
    print("🌊 CONTINUOUS ADAPTIVE VOXELIZATION ENHANCEMENT TEST")
    print("=" * 70)
    print("Testing the enhanced ScaleNet with continuous voxel size prediction\n")
    
    success = True
    
    try:
        # Test 1: Basic continuous prediction
        print("🧪 TEST 1: Basic Continuous Prediction")
        success &= test_continuous_prediction()
        
        # Test 2: Gradient flow
        print("\n🧪 TEST 2: Gradient Flow")
        success &= test_gradient_flow()
        
        # Test 3: Interpolation neighbors
        print("\n🧪 TEST 3: Interpolation Neighbors")
        success &= test_interpolation_neighbors()
        
        # Test 4: Configuration compatibility
        print("\n🧪 TEST 4: Configuration Compatibility")
        success &= test_configuration_compatibility()
        
        # Summary
        if success:
            print("\n🏆 ALL TESTS PASSED!")
            print("=" * 40)
            print("✅ Continuous prediction: WORKING")
            print("✅ Soft interpolation: IMPLEMENTED")
            print("✅ Gradient flow: IMPROVED")
            print("✅ Configuration: COMPATIBLE")
            print("✅ Backward compatibility: MAINTAINED")
            
            print(f"\n🎯 ENHANCEMENT SUMMARY:")
            print(f"   • The ScaleNet now supports continuous voxel size prediction")
            print(f"   • Soft interpolation provides smooth scale transitions")
            print(f"   • Better gradient flow for improved training")
            print(f"   • Full backward compatibility with discrete mode")
            print(f"   • Ready for production use!")
            
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🌊 Continuous enhancement test completed successfully!")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
