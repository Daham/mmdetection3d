#!/usr/bin/env python3
"""
Quick script to check if KITTI dataset is properly set up for training
"""

import os
import sys

def check_kitti_dataset(data_root):
    """Check if KITTI dataset is properly structured and has required files"""
    
    print(f"🔍 Checking KITTI dataset at: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"❌ Dataset root does not exist: {data_root}")
        return False
        
    print(f"✅ Dataset root exists: {data_root}")
    
    # Check required files
    required_files = [
        'kitti_infos_train.pkl',
        'kitti_infos_val.pkl', 
        'kitti_dbinfos_train.pkl'
    ]
    
    print("\n📁 Checking required preprocessed files:")
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(data_root, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    # Check data directories
    print("\n📂 Checking data directories:")
    data_dirs = [
        'training/velodyne',
        'training/velodyne_reduced', 
        'training/label_2',
        'training/calib'
    ]
    
    for dir_name in data_dirs:
        dir_path = os.path.join(data_root, dir_name)
        if os.path.exists(dir_path):
            num_files = len(os.listdir(dir_path)) if os.path.isdir(dir_path) else 0
            print(f"✅ {dir_name} ({num_files} files)")
        else:
            print(f"❌ {dir_name} - MISSING")
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files. You need to run data preparation.")
        print("🔧 To prepare KITTI data, run:")
        print("    python tools/create_data.py kitti --root-path /path/to/kitti --out-dir /path/to/kitti")
        return False
    else:
        print("\n🎉 All required files found! Dataset is ready for training.")
        return True

if __name__ == "__main__":
    # Use the data_root from config
    data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
    
    success = check_kitti_dataset(data_root)
    
    if not success:
        print("\n💡 Next steps:")
        print("1. Make sure KITTI dataset is downloaded and extracted")
        print("2. Run the data preparation script to generate .pkl files")
        print("3. Update the data_root path in the config if needed")
        sys.exit(1)
    else:
        print("\n🚀 Ready to start training!")
        sys.exit(0)
