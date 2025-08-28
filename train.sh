#!/bin/bash

# PhD Research Training Script - Novel Implementation
# Importance-Guided Multi-Scale VFE for 3D Object Detection
# Author: Daham

echo "🎯 Starting PhD Research Training - Novel Implementation"
echo "📊 Model: ImportanceGuidedMultiScaleVFE with Memory Optimization"
echo "🗂️  Dataset: KITTI"
echo "⚡ Features: Adaptive voxel scales, importance guidance, memory optimization"
echo "============================================================"

# Activate virtual environment
source /home/daham/mmdetection_project/mmdet_env/bin/activate

# Clean any previous cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Start training with your novel implementation
python tools/train.py configs/second/second_hv_secfpn_memory_optimized_kitti.py

echo "============================================================"
echo "🎓 PhD Research Training Complete - Novel Implementation"
