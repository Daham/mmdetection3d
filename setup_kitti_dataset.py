#!/usr/bin/env python3
"""
KITTI Dataset Setup Guide for Adaptive Voxelization Training

This script helps you set up the KITTI dataset for training the adaptive voxelization model.
Run this on your GPU machine where the dataset is located.
"""

import os
import sys

def check_and_prepare_kitti(data_root):
    """Check KITTI dataset and guide through preparation steps"""
    
    print("🎯 KITTI Dataset Setup for Adaptive Voxelization")
    print("=" * 60)
    print(f"📁 Dataset root: {data_root}")
    
    # Step 1: Check if data_root exists
    if not os.path.exists(data_root):
        print(f"\n❌ Error: Dataset path does not exist!")
        print(f"Expected: {data_root}")
        print("\n🔧 Solutions:")
        print("1. Create the directory if it doesn't exist:")
        print(f"   mkdir -p {data_root}")
        print("2. Download KITTI dataset from: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d")
        print("3. Extract the following files to your dataset directory:")
        print("   - training/velodyne/ (point clouds)")
        print("   - training/label_2/ (annotations)")  
        print("   - training/calib/ (calibration)")
        return False
    
    print(f"✅ Dataset root exists")
    
    # Step 2: Check raw data directories
    print(f"\n📂 Checking raw data directories...")
    raw_dirs = {
        'training/velodyne': 'Point cloud files (.bin)',
        'training/label_2': 'Annotation files (.txt)',
        'training/calib': 'Calibration files (.txt)'
    }
    
    missing_dirs = []
    for dir_name, description in raw_dirs.items():
        dir_path = os.path.join(data_root, dir_name)
        if os.path.exists(dir_path):
            num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"✅ {dir_name}: {num_files} files ({description})")
        else:
            print(f"❌ {dir_name}: Missing ({description})")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  Missing {len(missing_dirs)} required directories!")
        print("Please download and extract KITTI dataset properly.")
        return False
    
    # Step 3: Check preprocessed files
    print(f"\n📄 Checking preprocessed files...")
    required_files = [
        'kitti_infos_train.pkl',
        'kitti_infos_val.pkl',
        'kitti_dbinfos_train.pkl'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(data_root, file_name)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"✅ {file_name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {file_name}: Missing")
            missing_files.append(file_name)
    
    # Step 4: Generate missing preprocessed files
    if missing_files:
        print(f"\n🔧 Need to generate {len(missing_files)} preprocessed files")
        print("Run the following command in your MMDetection3D directory:")
        print(f"\n    python tools/create_data.py kitti \\")
        print(f"        --root-path {data_root} \\")
        print(f"        --out-dir {data_root} \\")
        print(f"        --extra-tag kitti")
        print(f"\nThis will create:")
        for file_name in missing_files:
            print(f"  - {file_name}")
        
        return False
    
    print(f"\n🎉 KITTI dataset is properly set up!")
    print(f"✅ All required files are present")
    print(f"🚀 Ready for adaptive voxelization training!")
    
    return True

def main():
    # Your dataset path
    data_root = '/home/daham/mmdetection_project/dataset/KITTI/'
    
    print("🔍 Adaptive Voxelization - KITTI Dataset Checker")
    print("This script verifies your KITTI dataset is ready for training.\n")
    
    success = check_and_prepare_kitti(data_root)
    
    if not success:
        print(f"\n💡 After setting up the dataset, update the config file:")
        print(f"   Edit: configs/second/adaptive_multiscale.py")
        print(f"   Set: data_root = '{data_root}'")
        print(f"\n🎯 Then you can start training with:")
        print(f"   python tools/train.py configs/second/adaptive_multiscale.py")
        sys.exit(1)
    else:
        print(f"\n🎯 Start training now:")
        print(f"   python tools/train.py configs/second/adaptive_multiscale.py")
        sys.exit(0)

if __name__ == "__main__":
    main()
