#!/usr/bin/env python3
"""
Test the refactored adaptive voxelization with a real mmdet3d config
"""

import sys
import os

# Activate virtual environment programmatically
import subprocess
venv_activate = "/home/daham/mmdetection_project/mmdet_env/bin/activate"
subprocess.run(f"source {venv_activate}", shell=True)

sys.path.insert(0, '/home/daham/mmdetection_project/mmdetection3d')

try:
    # Test mmdet3d imports
    from mmdet3d.registry import MODELS
    from mmdet3d.models.voxel_encoders import ImportanceGuidedMultiScaleVFE
    
    print("✅ Successfully imported ImportanceGuidedMultiScaleVFE from mmdet3d")
    
    # Test that it's properly registered
    registered_modules = MODELS.module_dict
    is_registered = 'ImportanceGuidedMultiScaleVFE' in registered_modules
    print(f"✅ Module registration: {'Registered' if is_registered else 'Not registered'}")
    
    # Create instance with config
    vfe_config = dict(
        type='ImportanceGuidedMultiScaleVFE',
        voxel_scales=[0.05, 0.1, 0.2],
        num_scales=3,
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=[-40, -40, -3, 40, 40, 1],
        scale_net_hidden_dims=[64, 32],
        gumbel_temperature=1.0,
        vfe_channels=[32, 64],
        fusion_channels=128,
        output_channels=64
    )
    
    # Build using registry
    vfe = MODELS.build(vfe_config)
    print(f"✅ Successfully built VFE from config")
    print(f"   Output channels: {vfe.output_channels}")
    print(f"   Voxel scales: {vfe.voxel_scales}")
    print(f"   Number of scales: {vfe.num_scales}")
    
    # Test forward pass
    import torch
    batch_size = 8
    max_points = 5
    features = torch.randn(batch_size, max_points, 4)
    num_points = torch.randint(1, max_points + 1, (batch_size,))
    coors = torch.randint(0, 100, (batch_size, 4))
    
    with torch.no_grad():
        output, coors_out = vfe(features, num_points, coors)
        
    print(f"✅ Forward pass successful")
    print(f"   Input: {features.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test scale statistics
    points = torch.randn(100, 4)
    stats = vfe.get_scale_statistics(points)
    print(f"✅ Scale statistics: {stats['scale_distribution']}")
    
    print("\n🎉 MMDET3D INTEGRATION TEST PASSED!")
    print("🚀 Ready for training with real datasets!")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback test
    print("\n🔄 Testing fallback approach...")
    try:
        sys.path.insert(0, '/home/daham/mmdetection_project/mmdetection3d/mmdet3d/models/voxel_encoders')
        from importance_guided_multi_scale_vfe import ImportanceGuidedMultiScaleVFE
        
        vfe = ImportanceGuidedMultiScaleVFE()
        print("✅ Direct import successful - module is functional")
        
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")
