#!/bin/bash
# 🎓 PhD RESEARCH: Training Command for Learnable Voxel Scale Parameters
# This script trains your adaptive voxelization model with learnable scale parameters

echo "🎓 PhD RESEARCH: Training Adaptive Voxelization with Learnable Scale Parameters"
echo "================================================================================"
echo "🎯 Model: SECOND with MemoryOptimizedImportanceGuidedMultiScaleVFE"
echo "🔬 Research: Learning optimal voxel sizes through backpropagation"
echo "📊 Dataset: KITTI 3D Car Detection"
echo "================================================================================"

# Activate virtual environment (CRITICAL!)
source /home/daham/mmdetection_project/mmdet_env/bin/activate

# Change to project directory
cd /home/daham/mmdetection_project/mmdetection3d

# Training command with learnable voxel scales
python tools/train.py \
    configs/second/second_hv_secfpn_memory_optimized_kitti.py \
    --work-dir work_dirs/learnable_voxel_scales_experiment \
    --seed 42 \
    --deterministic

echo "🎉 Training completed! Check work_dirs/learnable_voxel_scales_experiment for results"
echo "📈 Look for scale learning logs: 'LEARNABLE SCALES' and 'Scale gradients'"
