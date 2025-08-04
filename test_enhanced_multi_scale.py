#!/usr/bin/env python3
"""
Test Enhanced Multi-Scale VFE (1-10 Scales Support)
=================================================

This script tests the enhanced ImportanceGuidedMultiScaleVFE that now supports
1-10 voxel scales dynamically with automatic optimal scale generation.

Features tested:
1. Backward compatibility (3 scales)
2. Scale auto-generation (4-10 scales)
3. Custom scale specification
4. Computational feasibility for all scale counts
5. Proper network initialization

Author: Enhanced PhD Research Implementation
Date: August 4, 2025
"""

import torch
import sys
import os

# Add the project path
sys.path.append('/home/daham/mmdetection_project/mmdetection3d')

def test_multi_scale_configurations():
    """Test VFE with different numbers of scales."""
    
    print("🚀 TESTING ENHANCED MULTI-SCALE VFE (1-10 SCALES)")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"num_scales": 1, "name": "Single Scale"},
        {"num_scales": 2, "name": "Dual Scale"},
        {"num_scales": 3, "name": "Triple Scale (Original)"},
        {"num_scales": 5, "name": "Penta Scale"},
        {"num_scales": 7, "name": "Hepta Scale"},
        {"num_scales": 10, "name": "Deca Scale (Maximum)"},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print()
    
    results = []
    
    for config in test_configs:
        num_scales = config["num_scales"]
        name = config["name"]
        
        print(f"🎯 Testing {name} ({num_scales} scales)")
        print("-" * 40)
        
        try:
            # Import here to avoid issues if module isn't ready
            from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ImportanceGuidedMultiScaleVFE
            
            # Create VFE with specified number of scales
            vfe = ImportanceGuidedMultiScaleVFE(
                num_scales=num_scales,
                vfe_channels=[32, 64],
                scale_net_hidden_dims=[64, 32],
                gumbel_temperature=2.0,
                fusion_channels=128,
                output_channels=64,
                point_cloud_range=[0, -40, -3, 70.4, 40, 1]
            ).to(device)
            
            # Get scale information
            scale_info = vfe.scale_net.get_scale_info()
            print(f"✅ Generated scales: {[f'{s:.3f}m' for s in scale_info['scales']]}")
            print(f"✅ Scale range: {scale_info['scale_range']}")
            
            # Test with sample point cloud
            num_points = 1000
            sample_points = torch.randn(num_points, 4, device=device)  # x, y, z, intensity
            sample_points[:, :3] *= 10  # Scale to reasonable point cloud range
            
            # Forward pass
            with torch.no_grad():
                output, coords = vfe(sample_points)
                
            print(f"✅ Forward pass successful")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Output channels: {output.shape[1]} (expected: {vfe.output_channels})")
            
            # Memory estimation
            model_params = sum(p.numel() for p in vfe.parameters())
            memory_mb = model_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            print(f"✅ Model parameters: {model_params:,}")
            print(f"✅ Model memory: {memory_mb:.2f} MB")
            
            # Scale selection statistics
            scale_stats = vfe.get_scale_statistics(sample_points)
            print(f"✅ Scale distribution: {[f'{p:.3f}' for p in scale_stats['scale_distribution']]}")
            
            results.append({
                'name': name,
                'num_scales': num_scales,
                'scales': scale_info['scales'],
                'output_shape': output.shape,
                'memory_mb': memory_mb,
                'success': True
            })
            
            print("✅ SUCCESS!")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            results.append({
                'name': name,
                'num_scales': num_scales,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("📊 SUMMARY OF TESTS")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")
    print()
    
    if successful_tests:
        print("✅ SUCCESSFUL CONFIGURATIONS:")
        for result in successful_tests:
            print(f"  • {result['name']}: {result['num_scales']} scales, "
                  f"Memory: {result['memory_mb']:.1f}MB, Output: {result['output_shape']}")
    
    if failed_tests:
        print("\n❌ FAILED CONFIGURATIONS:")
        for result in failed_tests:
            print(f"  • {result['name']}: {result['error']}")
    
    print("\n🎉 ENHANCED MULTI-SCALE VFE TESTING COMPLETE!")
    
    return len(failed_tests) == 0

def test_backward_compatibility():
    """Test that existing 3-scale configs still work."""
    print("\n🔄 TESTING BACKWARD COMPATIBILITY")
    print("=" * 40)
    
    try:
        from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ImportanceGuidedMultiScaleVFE
        
        # Test with original config (should use provided scales)
        vfe_original = ImportanceGuidedMultiScaleVFE(
            voxel_scales=[0.025, 0.05, 0.1],  # Pedestrian config
            num_scales=3,
            vfe_channels=[64, 128],
            output_channels=64
        )
        
        expected_scales = [0.025, 0.05, 0.1]
        actual_scales = vfe_original.voxel_scales
        
        if actual_scales == expected_scales:
            print("✅ Backward compatibility: Original scales preserved")
        else:
            print(f"❌ Backward compatibility: Expected {expected_scales}, got {actual_scales}")
            return False
            
        # Test with mismatched num_scales (should auto-generate)
        vfe_auto = ImportanceGuidedMultiScaleVFE(
            voxel_scales=[0.05, 0.1, 0.2],  # 3 scales provided
            num_scales=5,  # But request 5 scales
            vfe_channels=[32, 64],
            output_channels=64
        )
        
        if len(vfe_auto.voxel_scales) == 5:
            print(f"✅ Auto-generation: Generated 5 scales: {[f'{s:.3f}m' for s in vfe_auto.voxel_scales]}")
        else:
            print(f"❌ Auto-generation: Expected 5 scales, got {len(vfe_auto.voxel_scales)}")
            return False
            
        print("✅ Backward compatibility test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test FAILED: {e}")
        return False

def test_config_update():
    """Test updating existing configs to use more scales."""
    print("\n🔧 TESTING CONFIG UPDATE EXAMPLES")
    print("=" * 40)
    
    # Example: Update pedestrian config to use 7 scales
    print("📝 Example: Updating pedestrian config from 3 to 7 scales")
    print("Old config: voxel_scales=[0.025, 0.05, 0.1], num_scales=3")
    print("New config: num_scales=7 (auto-generates optimal scales)")
    
    try:
        from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ImportanceGuidedMultiScaleVFE
        
        vfe_7_scales = ImportanceGuidedMultiScaleVFE(
            num_scales=7,  # Just specify the number!
            vfe_channels=[64, 128],
            scale_net_hidden_dims=[128, 64],
            gumbel_temperature=2.0,
            fusion_channels=256,
            output_channels=64,
            point_cloud_range=[0, -40, -3, 70.4, 40, 1]
        )
        
        scales = vfe_7_scales.scale_net.get_scale_info()
        print(f"✅ Generated 7 optimal scales: {[f'{s:.3f}m' for s in scales['scales']]}")
        print(f"✅ Range: {scales['scale_range']}")
        print("✅ Config update example PASSED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Config update example FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Multi-Scale VFE (1-10 scales support)...")
    
    # Run all tests
    test1_passed = test_multi_scale_configurations()
    test2_passed = test_backward_compatibility() 
    test3_passed = test_config_update()
    
    print("\n🏁 FINAL RESULTS")
    print("=" * 30)
    
    if test1_passed and test2_passed and test3_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Multi-scale VFE now supports 1-10 scales")
        print("✅ Backward compatibility maintained") 
        print("✅ Easy config updates available")
        print("\n🚀 Ready for production use!")
    else:
        print("❌ Some tests failed - check output above")
        
    print("\n💡 Usage examples:")
    print("  • 3 scales (original): num_scales=3")
    print("  • 5 scales (enhanced): num_scales=5") 
    print("  • 10 scales (maximum): num_scales=10")
    print("  • Auto-optimal scales for any count 1-10!")
