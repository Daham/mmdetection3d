#!/usr/bin/env python3
"""
Data availability checker for KITTI dataset
This will help identify data loading issues
"""

import os
import time

def check_data_availability():
    print("🔍 Checking KITTI data availability...")
    
    # Common data paths
    data_paths = [
        "data/kitti/",
        "./data/kitti/",
        "/data/kitti/",
        "../data/kitti/"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Found data directory: {path}")
            
            # Check for key files
            key_files = [
                "kitti_infos_train.pkl",
                "kitti_infos_val.pkl", 
                "training/velodyne_reduced/",
                "training/velodyne/"
            ]
            
            for key_file in key_files:
                full_path = os.path.join(path, key_file)
                if os.path.exists(full_path):
                    if os.path.isdir(full_path):
                        count = len(os.listdir(full_path))
                        print(f"   ✅ {key_file}: {count} files")
                    else:
                        size = os.path.getsize(full_path)
                        print(f"   ✅ {key_file}: {size} bytes")
                else:
                    print(f"   ❌ Missing: {key_file}")
            
            return path
    
    print("❌ No KITTI data directory found!")
    return None

def test_data_loading_speed():
    print("\n🚀 Testing data loading speed...")
    
    try:
        # Simple data loading test
        import torch
        from mmdet3d.datasets import KittiDataset
        
        # Minimal config for testing
        dataset_config = dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='kitti_infos_train.pkl',
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            ],
            test_mode=False
        )
        
        print("Creating dataset...")
        start_time = time.time()
        
        # This will fail if data is not available, which is good for debugging
        dataset = KittiDataset(**dataset_config)
        
        create_time = time.time() - start_time
        print(f"✅ Dataset created in {create_time:.2f} seconds")
        print(f"   Dataset length: {len(dataset)}")
        
        # Test loading first sample
        print("Loading first sample...")
        start_time = time.time()
        
        first_sample = dataset[0]
        
        load_time = time.time() - start_time
        print(f"✅ First sample loaded in {load_time:.2f} seconds")
        print(f"   Sample keys: {list(first_sample.keys())}")
        
        if load_time > 5.0:
            print("⚠️  WARNING: Data loading is very slow!")
            print("   This is likely the cause of your training slowdown.")
        else:
            print("✅ Data loading speed looks normal.")
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        print("   This confirms there's a data setup issue.")

def check_system_resources():
    print("\n💻 Checking system resources...")
    
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f"Memory usage: {memory.percent}% ({memory.used / 1e9:.1f}GB used / {memory.total / 1e9:.1f}GB total)")
        
        # Disk usage for current directory
        disk = psutil.disk_usage('.')
        print(f"Disk usage: {disk.percent}% ({disk.used / 1e9:.1f}GB used / {disk.total / 1e9:.1f}GB total)")
        
    except ImportError:
        print("psutil not available - install with: pip install psutil")
    except Exception as e:
        print(f"Resource check failed: {e}")

if __name__ == "__main__":
    print("🔧 DIAGNOSING SLOW TRAINING ISSUE")
    print("=" * 50)
    
    data_path = check_data_availability()
    
    if data_path:
        test_data_loading_speed()
    else:
        print("\n💡 SOLUTION: Set up KITTI dataset properly")
        print("   1. Download KITTI dataset")
        print("   2. Run data preparation scripts")
        print("   3. Ensure data is in the right location")
    
    check_system_resources()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
