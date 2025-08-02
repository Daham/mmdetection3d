#!/usr/bin/env python3
"""
Quick performance diagnostic to find bottlenecks
"""

import time
import os

def check_data_speed():
    """Check if data loading is slow"""
    print("🔍 Testing data loading speed...")
    
    data_paths = ['data/kitti/', './data/kitti/']
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            print(f"✅ Found data at: {data_path}")
            
            # Check if files exist
            train_file = os.path.join(data_path, 'kitti_infos_train.pkl')
            if os.path.exists(train_file):
                size = os.path.getsize(train_file)
                print(f"   Train file: {size / 1e6:.1f} MB")
                
                # Test file read speed
                start = time.time()
                with open(train_file, 'rb') as f:
                    data = f.read(1024 * 1024)  # Read 1MB
                read_time = time.time() - start
                print(f"   Read speed: {1.0/read_time:.1f} MB/s")
                
                if read_time > 1.0:
                    print("   ⚠️  WARNING: Slow disk I/O detected!")
            else:
                print(f"   ❌ Missing: {train_file}")
            break
    else:
        print("❌ No data directory found")

def check_import_speed():
    """Check if imports are slow"""
    print("\n🔍 Testing import speed...")
    
    imports_to_test = [
        ('torch', 'import torch'),
        ('mmdet3d', 'import mmdet3d'),
        ('mmcv', 'import mmcv'),
        ('mmengine', 'import mmengine')
    ]
    
    for name, import_cmd in imports_to_test:
        start = time.time()
        try:
            exec(import_cmd)
            import_time = time.time() - start
            print(f"   {name}: {import_time:.2f}s")
            
            if import_time > 5.0:
                print(f"      ⚠️  WARNING: {name} import is very slow!")
        except ImportError as e:
            print(f"   ❌ {name}: Failed - {e}")

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Testing GPU...")
    
    try:
        import torch
        
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            
            # Test GPU speed
            start = time.time()
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"   GPU compute test: {gpu_time:.3f}s")
            
            if gpu_time > 1.0:
                print("      ⚠️  WARNING: GPU is slow!")
        else:
            print("   ⚠️  No GPU available - training will be very slow on CPU!")
            
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")

def quick_model_test():
    """Test model creation speed"""
    print("\n🔍 Testing model creation...")
    
    try:
        start = time.time()
        
        # Test creating a simple model
        from mmdet3d.models import build_detector
        from mmengine.config import Config
        
        # Minimal config
        model_cfg = dict(
            type='VoxelNet',
            voxel_encoder=dict(type='HardSimpleVFE'),
            middle_encoder=dict(
                type='SparseEncoder',
                in_channels=4,
                sparse_shape=[41, 160, 140]),
            backbone=dict(
                type='SECOND',
                in_channels=256,
                layer_nums=[1],
                layer_strides=[1],
                out_channels=[64]),
            neck=dict(
                type='SECONDFPN',
                in_channels=[64],
                upsample_strides=[1],
                out_channels=[128]),
            bbox_head=dict(
                type='Anchor3DHead',
                num_classes=1,
                in_channels=128,
                feat_channels=64))
        
        model = build_detector(model_cfg)
        
        model_time = time.time() - start
        print(f"   Model creation: {model_time:.2f}s")
        
        if model_time > 10.0:
            print("      ⚠️  WARNING: Model creation is very slow!")
        else:
            print("   ✅ Model creation speed OK")
            
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")

if __name__ == "__main__":
    print("🚀 PERFORMANCE DIAGNOSTIC")
    print("=" * 40)
    
    check_import_speed()
    check_gpu()
    check_data_speed()
    quick_model_test()
    
    print("\n" + "=" * 40)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("\n💡 Common fixes for slow training:")
    print("   1. Use GPU instead of CPU")
    print("   2. Reduce batch size")
    print("   3. Use faster storage (SSD)")
    print("   4. Reduce model complexity")
    print("   5. Check dataset size/preprocessing")
