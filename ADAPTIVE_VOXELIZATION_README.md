# Adaptive Voxelization Research Implementation

This repository contains a research-grade implementation of learnable adaptive voxelization for 3D object detection using MMDetection3D.

## 🎓 Research Features

- **Learnable Voxel Sizes**: Neural networks predict optimal voxel sizes for different spatial regions
- **Multi-Scale Processing**: 4 parallel pathways process different voxel size groups independently
- **Attention-Based Fusion**: Learned attention mechanisms combine multi-scale features
- **End-to-End Training**: Full pipeline is differentiable and GPU-optimized
- **Memory Efficient**: Sparse convolution maintains computational efficiency

## 📁 Core Implementation Files

### Models
- `mmdet3d/models/voxel_encoders/adaptive_sparse_bridge.py` - Learnable adaptive voxel encoder
- `mmdet3d/models/middle_encoders/adaptive_sparse_encoder.py` - Multi-scale sparse convolution encoder
- `mmdet3d/models/detectors/adaptive_voxelnet.py` - Custom detector for adaptive pipeline

### Configuration
- `configs/second/adaptive_multiscale.py` - Main standalone configuration file

## 🚀 Usage

### Setup
1. Update the `data_root` path in `configs/second/adaptive_multiscale.py` to point to your KITTI dataset
2. Ensure you have a GPU environment (sparse convolution requires CUDA)

### Training
```bash
python tools/train.py configs/second/adaptive_multiscale.py
```

### Testing
```bash
python tools/test.py configs/second/adaptive_multiscale.py work_dirs/adaptive_multiscale/latest.pth
```

## 🔬 Research Components

### Adaptive Voxel Encoder (AdaptiveSparseBridge)
- Learns voxel sizes based on spatial features
- Predicts sizes in range [0.05, 0.50] meters
- Uses neural networks for size prediction

### Multi-Scale Sparse Encoder (AdaptiveSparseEncoder)
- 4 parallel processing pathways for different voxel sizes:
  - Fine (0.05-0.15m): Small objects, detailed features
  - Medium-Fine (0.15-0.25m): Medium objects
  - Medium (0.25-0.35m): Cars, trucks
  - Coarse (0.35-0.50m): Large structures
- Attention-based fusion of pathway outputs
- LayerNorm for training stability

### Custom Detector (AdaptiveVoxelNet)
- Handles learned voxel sizes throughout pipeline
- Maintains compatibility with standard MMDetection3D components
- Research logging for voxel size analysis

## 🎯 Research Applications

This implementation enables research on:
- How learned voxel sizes improve detection accuracy
- What size patterns emerge for different object types  
- Multi-scale feature fusion strategies
- Computational efficiency of size-specific pathways
- Adaptive voxelization for other 3D tasks

## 📊 Training Configuration

- Enhanced learning rates for adaptive components
- Cosine annealing schedule optimized for voxel size learning
- Memory-efficient batch processing
- GPU-optimized sparse convolution operations

## 🔧 Requirements

- CUDA-capable GPU (sparse convolution requirement)
- MMDetection3D framework
- spconv library
- PyTorch with CUDA support

## 📝 Citation

If you use this implementation in your research, please cite the relevant papers and acknowledge the adaptive voxelization methodology.
