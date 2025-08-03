#!/usr/bin/env python3

"""
Test script for refactored adaptive voxelization configuration
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/daham/mmdetection_project/mmdetection3d')

def test_configuration():
    """Test the refactored configuration"""
    print("🔬 TESTING REFACTORED CONFIGURATION")
    print("="*60)
    
    try:
        print("🔧 Testing refactored configuration...")
        
        # Import necessary modules
        from mmengine.config import Config
        import mmdet3d.models.voxel_encoders  # This triggers registration
        
        # Load configuration
        config_path = '/home/daham/mmdetection_project/mmdetection3d/configs/second/optimized_adaptive_multi_scale.py'
        cfg = Config.fromfile(config_path)
        
        # Check VFE configuration
        vfe_cfg = cfg.model.voxel_encoder
        print("✅ Configuration loaded successfully")
        print(f"   VFE Type: {vfe_cfg.type}")
        print(f"   Voxel Scales: {vfe_cfg.voxel_scales}")
        print(f"   Output Channels: {vfe_cfg.output_channels}")
        print(f"   Work Dir: {cfg.work_dir}")
        
        # Check if VFE is available
        from mmdet3d.registry import MODELS
        available_vfes = [k for k in MODELS.module_dict.keys() if 'VFE' in k]
        
        if 'ImportanceGuidedMultiScaleVFE' in MODELS.module_dict:
            print("✅ ImportanceGuidedMultiScaleVFE found in registry")
        else:
            print("❌ ImportanceGuidedMultiScaleVFE not found in registry")
            print(f"Available VFEs: {available_vfes}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset configuration"""
    print("\n🔧 Testing dataset loading...")
    
    try:
        from mmengine.config import Config
        
        config_path = '/home/daham/mmdetection_project/mmdetection3d/configs/second/optimized_adaptive_multi_scale.py'
        cfg = Config.fromfile(config_path)
        
        # Check dataset configuration
        if hasattr(cfg, 'train_dataloader'):
            print("✅ Train dataloader configuration found")
        else:
            print("❌ Train dataloader configuration missing")
            return False
            
        # Check if data directory exists (optional)
        data_root = getattr(cfg, 'data_root', '/home/daham/mmdetection_project/mmdetection3d/data/kitti')
        if os.path.exists(data_root):
            print(f"✅ Data directory found: {data_root}")
        else:
            print(f"⚠️ KITTI data directory not found: {data_root}")
            print("   This is expected if KITTI dataset is not downloaded")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

if __name__ == "__main__":
    config_ok = test_configuration()
    dataset_ok = test_dataset()
    
    if config_ok and dataset_ok:
        print("\n✅ All tests passed - Configuration ready for training")
        sys.exit(0)
    else:
        print("\n❌ Configuration has issues - needs fixing before training")
        sys.exit(1)
