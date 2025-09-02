# 🎓 PhD Research: Adaptive Voxelization Experimental Results

## 📊 Table I: 3D Detection AP (%) on KITTI Validation Set for Car Category (IoU=0.7)

### Experimental Setup
- **Dataset**: KITTI 3D Object Detection (Car category)
- **Training Epochs**: 2 epochs per experiment
- **Hardware**: NVIDIA RTX 4070 Super (12GB VRAM)
- **Optimizer**: AdamW (lr=0.003, weight_decay=0.01)
- **Mixed Precision**: Enabled (AMP)
- **Date**: September 2, 2025

---

## 🚀 Experimental Results

### ✅ Experiment 1: Fixed Single Scale (0.1m voxels)
**Configuration**: `configs/second/second_fixed_single_scale_baseline_kitti.py`
**Status**: Completed
**Training Started**: 21:35:16 (Sept 2, 2025)
**Training Performance**: 
- Initial warmup: ~0.37s per batch (steps 1-50)
- Stable training: ~0.26s per batch (steps 50+)
- Total batches per epoch: 1667
- Time per epoch: ~7.2 minutes
- **Total training time**: ~14.4 minutes (2 epochs)
- Loss progression: 4.85 → 1.58 → 1.11 (good convergence)
- Memory usage: ~2.8GB (efficient on RTX 4070 Super)

#### 3D Detection AP@0.70 (IoU=0.7):
| Difficulty | AP11 (%) | AP40 (%) |
|------------|----------|----------|
| Easy       | 68.31    | 67.84    |
| Moderate   | 57.21    | 56.56    |
| Hard       | 53.50    | 52.18    |

#### Additional Metrics:
- **BEV AP@0.70**: Easy=86.58%, Moderate=77.61%, Hard=76.21%
- **2D AP@0.70**: Easy=88.30%, Moderate=81.57%, Hard=77.69%

#### Evaluation Completed:
- **Test completed**: 22:27:27 (Sept 2, 2025)
- **Test time**: ~3.3 minutes (5001 samples, 0.0394s per sample)
- **Final confirmed results** ✅

#### Full Results Log:
```
----------- AP11 Results ------------
Car AP11@0.70, 0.70, 0.70:
bbox AP11:88.2982, 81.5715, 77.6888
bev  AP11:86.5844, 77.6096, 76.2131
3d   AP11:68.3061, 57.2098, 53.5020
aos  AP11:87.67, 80.45, 76.39

Car AP11@0.70, 0.50, 0.50:
bbox AP11:88.2982, 81.5715, 77.6888
bev  AP11:89.7561, 88.6789, 87.2697
3d   AP11:89.7168, 88.2035, 84.9245
aos  AP11:87.67, 80.45, 76.39

----------- AP40 Results ------------
Car AP40@0.70, 0.70, 0.70:
bbox AP40:93.3340, 83.7831, 80.4209
bev  AP40:87.5028, 80.6433, 75.8218
3d   AP40:67.8365, 56.5648, 52.1763
aos  AP40:92.61, 82.56, 78.92

Car AP40@0.70, 0.50, 0.50:
bbox AP40:93.3340, 83.7831, 80.4209
bev  AP40:95.2745, 90.9726, 88.1408
3d   AP40:95.1475, 90.5428, 87.2038
aos  AP40:92.61, 82.56, 78.92
```

---

### 🚧 Experiment 2: Fixed Multi-Scale [0.05, 0.1, 0.2]m
**Configuration**: `configs/second/second_fixed_multiscale_baseline_kitti.py`
**Status**: Pending

#### 3D Detection AP@0.70 (IoU=0.7):
| Difficulty | AP11 (%) | AP40 (%) |
|------------|----------|----------|
| Easy       | -        | -        |
| Moderate   | -        | -        |
| Hard       | -        | -        |

---

### 🚧 Experiment 3: Random Scale Selection
**Configuration**: `configs/second/second_random_scale_baseline_kitti.py`
**Status**: Pending

#### 3D Detection AP@0.70 (IoU=0.7):
| Difficulty | AP11 (%) | AP40 (%) |
|------------|----------|----------|
| Easy       | -        | -        |
| Moderate   | -        | -        |
| Hard       | -        | -        |

---

### 🚧 Experiment 4: Single Learnable Scale
**Configuration**: `configs/second/second_single_learnable_scale_baseline_kitti.py`
**Status**: Pending

#### 3D Detection AP@0.70 (IoU=0.7):
| Difficulty | AP11 (%) | AP40 (%) |
|------------|----------|----------|
| Easy       | -        | -        |
| Moderate   | -        | -        |
| Hard       | -        | -        |

---

### 🚧 Experiment 5: Adaptive Voxelization (PhD Research Method)
**Configuration**: `configs/second/second_adaptive_voxelization_kitti.py`
**Status**: Pending

#### 3D Detection AP@0.70 (IoU=0.7):
| Difficulty | AP11 (%) | AP40 (%) |
|------------|----------|----------|
| Easy       | -        | -        |
| Moderate   | -        | -        |
| Hard       | -        | -        |

---

## 📈 Performance Analysis

### Current Observations:
1. **Experiment 1 Results (Fixed Single Scale 0.1m)**: 
   - Easy: 68.31%, Moderate: 57.21%, Hard: 53.50%
   - Strong baseline performance with 0.1m voxel size
   - Good detection across all difficulty levels

2. **Training Efficiency**: 2 epochs showing good convergence for evaluation

### Next Steps:
- Complete remaining 4 experiments
- Compare multi-scale approaches against single scale
- Analyze adaptive voxelization performance
- Compare computational overhead and memory usage

---

## 🔧 Technical Details

### Hardware Configuration:
- **GPU**: NVIDIA RTX 4070 Super (12GB VRAM)
- **Memory Optimization**: Mixed precision training (AMP)
- **Gradient Checkpointing**: Enabled for memory efficiency

### Training Configuration:
- **Optimizer**: AdamW
- **Learning Rate**: 0.003
- **Weight Decay**: 0.01
- **Epochs**: 2 per experiment
- **Batch Size**: Adaptive based on memory optimization

### Dataset Information:
- **Training Samples**: ~7,463 velodyne files
- **Validation**: KITTI standard split
- **Point Cloud Range**: [0, -40, -3, 70.4, 40, 1]
- **Target Class**: Car only

---

*Last Updated: September 2, 2025*
