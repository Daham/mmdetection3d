#!/usr/bin/env python3
"""
Training Script for Multi-Resolution Adaptive Voxelization

This script provides commands and guidance for training the multi-resolution adaptive model.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_environment():
    """Check if the environment is properly set up."""
    print("Checking Environment...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('mmdet3d'):
        print("❌ Not in MMDetection3D root directory")
        print("Please run this script from the MMDetection3D root directory")
        return False
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   PyTorch version: {torch.__version__}")
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check if spconv is available
    try:
        import spconv
        print(f"✅ spconv available: {spconv.__version__}")
    except ImportError:
        print("❌ spconv not installed")
        print("Install with: pip install spconv-cu118  # or appropriate CUDA version")
        return False
    
    # Check if MMDetection3D is properly installed
    try:
        import mmdet3d
        print(f"✅ MMDetection3D available: {mmdet3d.__version__}")
    except ImportError:
        print("❌ MMDetection3D not properly installed")
        return False
    
    # Check if our custom modules can be imported
    try:
        from mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe import EnhancedAdaptiveVFE
        from mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder import MultiResolutionSparseEncoder
        print("✅ Custom adaptive modules can be imported")
    except ImportError as e:
        print(f"❌ Cannot import custom modules: {e}")
        return False
    
    print("\n✅ Environment check passed!")
    return True


def validate_config():
    """Validate the training configuration."""
    print("\nValidating Configuration...")
    print("=" * 50)
    
    config_path = "configs/second/adaptive_multi_resolution_training.py"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        # Use MMDetection3D's config loading
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        
        print("✅ Config file loads successfully")
        print(f"   Model type: {cfg.model.type}")
        print(f"   VFE type: {cfg.model.voxel_encoder.type}")
        print(f"   Middle encoder type: {cfg.model.middle_encoder.type}")
        print(f"   Number of classes: {cfg.model.bbox_head.num_classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False


def run_training():
    """Run the training process."""
    print("\nStarting Training...")
    print("=" * 50)
    
    config_path = "configs/second/adaptive_multi_resolution_training.py"
    work_dir = "work_dirs/adaptive_multi_resolution"
    
    # Create work directory
    os.makedirs(work_dir, exist_ok=True)
    
    # Training command
    cmd = [
        "python", "tools/train.py",
        config_path,
        "--work-dir", work_dir,
        "--auto-scale-lr"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Work directory: {work_dir}")
    print(f"Logs will be saved to: {work_dir}/")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False


def run_testing():
    """Run testing on the trained model."""
    print("\nRunning Testing...")
    print("=" * 50)
    
    config_path = "configs/second/adaptive_multi_resolution_training.py"
    work_dir = "work_dirs/adaptive_multi_resolution"
    checkpoint_path = f"{work_dir}/latest.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first or specify a different checkpoint path")
        return False
    
    cmd = [
        "python", "tools/test.py",
        config_path,
        checkpoint_path,
        "--work-dir", work_dir,
        "--show-dir", f"{work_dir}/results"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Testing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Testing failed with error code {e.returncode}")
        return False


def main():
    """Main function to orchestrate the training process."""
    print("Multi-Resolution Adaptive Voxelization Training")
    print("=" * 60)
    print()
    print("This script will train a 3D object detection model using")
    print("multi-resolution adaptive voxelization.")
    print()
    
    # Step 1: Environment check
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Step 2: Config validation
    if not validate_config():
        print("\n❌ Configuration validation failed. Please check the config file.")
        sys.exit(1)
    
    # Step 3: Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Run training")
    print("2. Run testing (requires trained model)")
    print("3. Both training and testing")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            success = run_training()
            break
        elif choice == "2":
            success = run_testing()
            break
        elif choice == "3":
            success = run_training()
            if success:
                print("\nTraining completed, now running testing...")
                success = run_testing()
            break
        elif choice == "4":
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    if success:
        print("\n🎉 All operations completed successfully!")
        print("\nResults can be found in: work_dirs/adaptive_multi_resolution/")
        print("\nKey files:")
        print("  - Logs: work_dirs/adaptive_multi_resolution/*.log")
        print("  - Checkpoints: work_dirs/adaptive_multi_resolution/*.pth")
        print("  - Test results: work_dirs/adaptive_multi_resolution/results/")
    else:
        print("\n❌ Some operations failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
