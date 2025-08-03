#!/usr/bin/env python3

"""
Test script to check if ImportanceGuidedMultiScaleVFE can be imported and registered
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/daham/mmdetection_project/mmdetection3d')

try:
    print("🔧 Testing imports...")
    
    # Test basic mmdet3d imports
    print("   - Importing mmdet3d...")
    import mmdet3d
    
    print("   - Importing MODELS registry...")
    from mmdet3d.registry import MODELS
    
    print("   - Importing voxel encoders...")
    import mmdet3d.models.voxel_encoders
    
    print("   - Checking available VFEs in registry...")
    available_vfes = []
    for name, module in MODELS.module_dict.items():
        if 'VFE' in name or 'vfe' in name.lower():
            available_vfes.append(name)
    
    print(f"   Available VFEs: {available_vfes}")
    
    # Test specific import
    print("   - Testing ImportanceGuidedMultiScaleVFE import...")
    from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ImportanceGuidedMultiScaleVFE
    
    print("   - Testing registry lookup...")
    if 'ImportanceGuidedMultiScaleVFE' in MODELS.module_dict:
        print("   ✅ ImportanceGuidedMultiScaleVFE found in registry!")
    else:
        print("   ❌ ImportanceGuidedMultiScaleVFE NOT found in registry")
        print(f"   Available modules: {list(MODELS.module_dict.keys())}")
    
    print("✅ Import test completed successfully")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
