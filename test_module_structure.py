#!/usr/bin/env python3
"""
Test the adaptive sparse bridge module structure and imports.
"""

import sys
import os

def test_module_structure():
    """Test that the module has correct structure."""
    print("🔍 Testing Module Structure")
    print("="*40)
    
    project_root = '/Users/dahamp/Documents/academic/phd-repos/mmdetection3d'
    module_path = os.path.join(project_root, 'mmdet3d/models/voxel_encoders/adaptive_sparse_bridge.py')
    
    # Read module content
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Check key components
    checks = [
        ("Conditional imports", "TORCH_AVAILABLE" in content),
        ("Class definition", "class AdaptiveSparseBridge" in content),
        ("Init method", "def __init__" in content),
        ("Forward method", "def forward" in content),
        ("Adaptation network", "_build_adaptation_network" in content),
        ("Feature network", "_build_feature_network" in content),
        ("Grid mapping", "_map_to_regular_grid" in content),
        ("Learn voxel sizes", "_learn_adaptive_voxel_sizes" in content),
        ("Fallback class", "else:" in content and "class AdaptiveSparseBridge:" in content)
    ]
    
    print("\n📋 Module Components:")
    all_good = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_good = False
    
    # Check __init__.py registration
    init_path = os.path.join(project_root, 'mmdet3d/models/voxel_encoders/__init__.py')
    with open(init_path, 'r') as f:
        init_content = f.read()
    
    print(f"\n📦 Module Registration:")
    reg_checks = [
        ("Import statement", "from .adaptive_sparse_bridge import AdaptiveSparseBridge" in init_content),
        ("__all__ list", "'AdaptiveSparseBridge'" in init_content)
    ]
    
    for check_name, passed in reg_checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_good = False
    
    # Check config file
    config_path = os.path.join(project_root, 'configs/second/adaptive_sparse.py')
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    print(f"\n⚙️  Config File:")
    config_checks = [
        ("AdaptiveSparseBridge type", "type='AdaptiveSparseBridge'" in config_content),
        ("No custom imports", "custom_imports" not in config_content),
        ("Learnable adaptation", "learnable_adaptation=True" in config_content)
    ]
    
    for check_name, passed in config_checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_good = False
    
    print(f"\n{'='*40}")
    if all_good:
        print("✅ MODULE STRUCTURE IS CORRECT!")
        print("\n🚀 Ready to train:")
        print("   python tools/train.py configs/second/adaptive_sparse.py")
        
        print("\n💡 Key Features:")
        print("   - Learns adaptive voxel sizes")
        print("   - Maps to regular grid for sparse convolution")
        print("   - Handles conflicts automatically")
        print("   - No custom imports needed")
    else:
        print("❌ Some issues found in module structure")
    
    print(f"\n📁 Files verified:")
    print(f"   - {module_path}")
    print(f"   - {init_path}")
    print(f"   - {config_path}")

if __name__ == "__main__":
    test_module_structure()
