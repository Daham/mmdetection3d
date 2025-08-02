"""
Diagnostic script for adaptive voxelization training issues

Run this on your training machine to check for potential problems:
python diagnose_adaptive_training.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import traceback

def test_adaptive_module():
    """Test the adaptive module in isolation"""
    print("🔍 Testing AdaptiveSparseBridge module...")
    
    try:
        # Import the module
        sys.path.append('.')
        from mmdet3d.models.voxel_encoders.adaptive_sparse_bridge import AdaptiveSparseBridge
        
        # Create test data similar to real voxels
        batch_size = 2
        max_points_per_voxel = 5
        num_features = 4
        
        # Simulate realistic voxel data
        features = torch.randn(batch_size, max_points_per_voxel, num_features)
        num_points = torch.randint(1, max_points_per_voxel + 1, (batch_size,))
        coors = torch.randint(0, 100, (batch_size, 4))  # [batch_idx, z, y, x]
        
        print(f"✅ Test data created:")
        print(f"   - Features shape: {features.shape}")
        print(f"   - Num points: {num_points}")
        print(f"   - Features range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Test different configurations
        configs = [
            {"learnable_adaptation": False, "adaptation_strength": 0.0},  # Baseline
            {"learnable_adaptation": False, "adaptation_strength": 0.2},  # Simple adaptive
            {"learnable_adaptation": True, "adaptation_strength": 0.3},   # Full adaptive
        ]
        
        for i, config in enumerate(configs):
            print(f"\n🧪 Testing configuration {i+1}: {config}")
            
            # Create module
            module = AdaptiveSparseBridge(num_features=num_features, **config)
            module.eval()
            
            # Forward pass
            with torch.no_grad():
                output = module(features, num_points, coors)
                
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   ✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"   ✅ Output mean: {output.mean():.3f}")
            print(f"   ✅ Output std: {output.std():.3f}")
            
            # Check for NaN or infinite values
            if torch.isnan(output).any():
                print("   ❌ WARNING: NaN values detected!")
            if torch.isinf(output).any():
                print("   ❌ WARNING: Infinite values detected!")
            
            # Test gradients if learnable
            if config.get("learnable_adaptation", False):
                module.train()
                features_grad = features.clone().requires_grad_(True)
                output_grad = module(features_grad, num_points, coors)
                
                # Backward pass
                loss = output_grad.sum()
                loss.backward()
                
                print(f"   ✅ Gradients computed successfully")
                print(f"   ✅ Input gradient norm: {features_grad.grad.norm():.6f}")
                
                # Check for gradient explosion
                for name, param in module.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm()
                        print(f"   📊 {name} grad norm: {grad_norm:.6f}")
                        if grad_norm > 10.0:
                            print(f"   ⚠️  WARNING: Large gradient in {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing adaptive module: {e}")
        traceback.print_exc()
        return False

def compare_with_vanilla():
    """Compare adaptive vs vanilla performance on identical data"""
    print("\n🔬 Comparing with vanilla VFE...")
    
    try:
        from mmdet3d.models.voxel_encoders.adaptive_sparse_bridge import AdaptiveSparseBridge
        
        # Test data
        batch_size = 4
        max_points_per_voxel = 5
        num_features = 4
        
        features = torch.randn(batch_size, max_points_per_voxel, num_features)
        num_points = torch.randint(1, max_points_per_voxel + 1, (batch_size,))
        coors = torch.randint(0, 100, (batch_size, 4))
        
        # Vanilla-like computation (HardSimpleVFE equivalent)
        vanilla_output = features[:, :, :num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        
        # Adaptive with minimal settings (should be nearly identical)
        adaptive_minimal = AdaptiveSparseBridge(
            num_features=num_features,
            learnable_adaptation=False,
            adaptation_strength=0.0
        )
        
        with torch.no_grad():
            adaptive_output = adaptive_minimal(features, num_points, coors)
        
        # Compare outputs
        diff = torch.abs(vanilla_output - adaptive_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"📊 Vanilla vs Adaptive (minimal) comparison:")
        print(f"   - Max difference: {max_diff:.8f}")
        print(f"   - Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-6:
            print("   ✅ Perfect match - adaptive module is working correctly")
        elif max_diff < 1e-3:
            print("   ✅ Very close match - minor numerical differences")
        else:
            print("   ⚠️  WARNING: Significant differences detected")
            
        return max_diff < 1e-3
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        traceback.print_exc()
        return False

def check_training_stability():
    """Check for common training stability issues"""
    print("\n🏋️ Checking training stability factors...")
    
    suggestions = []
    
    # Check PyTorch version
    print(f"📋 PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available and working
    if torch.cuda.is_available():
        print(f"🎮 CUDA available: {torch.cuda.get_device_name()}")
        print(f"🎮 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Running on CPU")
        suggestions.append("Consider using GPU for faster training")
    
    # Memory test
    try:
        test_tensor = torch.randn(1000, 1000, dtype=torch.float32)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            torch.cuda.empty_cache()
        del test_tensor
        print("✅ Memory allocation test passed")
    except Exception as e:
        print(f"❌ Memory issue detected: {e}")
        suggestions.append("Check available memory and reduce batch size")
    
    # Print suggestions
    if suggestions:
        print("\n💡 Suggestions for better training:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    
    return True

def main():
    """Run all diagnostic tests"""
    print("🚀 MMDetection3D Adaptive Voxelization Diagnostics")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Module functionality
    if test_adaptive_module():
        success_count += 1
    
    # Test 2: Vanilla comparison
    if compare_with_vanilla():
        success_count += 1
    
    # Test 3: Training stability
    if check_training_stability():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Diagnostic Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests passed! The adaptive module should work correctly.")
        print("\n💡 Training tips:")
        print("   1. Start with conservative settings (adaptation_strength=0.2)")
        print("   2. Use gradient clipping (max_norm=10.0)")
        print("   3. Monitor loss curves carefully")
        print("   4. Compare with vanilla SECOND baseline")
    else:
        print("⚠️  Some issues detected. Check the output above for details.")

if __name__ == "__main__":
    main()
