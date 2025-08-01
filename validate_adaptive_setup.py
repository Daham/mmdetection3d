#!/usr/bin/env python3
"""
Simple validation script to check if the adaptive modules can be imported and instantiated.
This script doesn't require heavy dependencies and focuses on basic functionality.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all custom modules can be imported."""
    print("Testing module imports...")
    print("-" * 40)
    
    try:
        from mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe import EnhancedAdaptiveVFE
        print("✓ EnhancedAdaptiveVFE imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import EnhancedAdaptiveVFE: {e}")
        return False
    
    try:
        from mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder import MultiResolutionSparseEncoder
        print("✓ MultiResolutionSparseEncoder imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MultiResolutionSparseEncoder: {e}")
        return False
    
    return True

def test_module_instantiation():
    """Test if modules can be instantiated with basic parameters."""
    print("\nTesting module instantiation...")
    print("-" * 40)
    
    try:
        from mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe import EnhancedAdaptiveVFE
        
        vfe = EnhancedAdaptiveVFE(
            in_channels=4,
            feat_channels=[64, 128],
            with_distance=True,
            voxel_size=(0.05, 0.05, 0.1),
            point_cloud_range=(0, -40, -3, 70.4, 40, 1),
            base_sparse_shape=[41, 1600, 1408],
            adaptation_method='multi_scale',
            num_scales=3,
            provide_multi_res_info=True
        )
        print("✓ EnhancedAdaptiveVFE instantiated successfully")
        
    except Exception as e:
        print(f"✗ Failed to instantiate EnhancedAdaptiveVFE: {e}")
        return False
    
    try:
        from mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder import MultiResolutionSparseEncoder
        
        encoder = MultiResolutionSparseEncoder(
            base_voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            resolution_levels=[0.5, 1.0, 2.0],
            in_channels=128,
            out_channels=256,
            assignment_threshold=0.1,
            fusion_method='attention'
        )
        print("✓ MultiResolutionSparseEncoder instantiated successfully")
        
    except Exception as e:
        print(f"✗ Failed to instantiate MultiResolutionSparseEncoder: {e}")
        return False
    
    return True

def test_config_loading():
    """Test if the training config can be loaded."""
    print("\nTesting configuration loading...")
    print("-" * 40)
    
    config_path = "configs/second/adaptive_multi_resolution_training.py"
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        # Try to load as a Python module
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        print("✓ Configuration file loads as Python module")
        
        # Check if it has the expected attributes
        if hasattr(config_module, 'model'):
            print("✓ Model configuration found")
        if hasattr(config_module, 'custom_imports'):
            print("✓ Custom imports configuration found")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False

def test_registry():
    """Test if modules are properly registered."""
    print("\nTesting module registry...")
    print("-" * 40)
    
    try:
        from mmdet3d.registry import MODELS
        
        # Check if our modules are in the registry
        if 'EnhancedAdaptiveVFE' in MODELS._module_dict:
            print("✓ EnhancedAdaptiveVFE registered in MODELS")
        else:
            print("✗ EnhancedAdaptiveVFE not found in MODELS registry")
            return False
        
        if 'MultiResolutionSparseEncoder' in MODELS._module_dict:
            print("✓ MultiResolutionSparseEncoder registered in MODELS")
        else:
            print("✗ MultiResolutionSparseEncoder not found in MODELS registry")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to check registry: {e}")
        return False

def test_file_structure():
    """Test if all required files exist."""
    print("\nTesting file structure...")
    print("-" * 40)
    
    required_files = [
        "mmdet3d/models/voxel_encoders/enhanced_adaptive_vfe.py",
        "mmdet3d/models/middle_encoders/multi_resolution_sparse_encoder.py",
        "configs/second/adaptive_multi_resolution_training.py",
        "ADAPTIVE_VOXELIZATION_README.md",
        "run_adaptive_training.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Run all validation tests."""
    print("Adaptive Voxelization Validation Script")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Module Instantiation", test_module_instantiation),
        ("Module Registry", test_registry),
        ("Configuration Loading", test_config_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = 0
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All validation tests passed!")
        print("The adaptive voxelization modules are ready for use.")
        print("\nNext steps:")
        print("1. Set up your dataset (KITTI)")
        print("2. Run training: python run_adaptive_training.py")
        print("3. Check results in work_dirs/adaptive_multi_resolution/")
    else:
        print("\n⚠️  Some validation tests failed.")
        print("Please check the error messages above and fix any issues.")
        print("Refer to ADAPTIVE_VOXELIZATION_README.md for detailed instructions.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
