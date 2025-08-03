# Benchmarking Configuration Comparison

## Purpose
Fair comparison between **Adaptive Multi-Scale VFE** (PhD research) and **Vanilla SECOND** baseline.

## Configuration Files

### 1. Adaptive Multi-Scale VFE (PhD Research)
**File**: `configs/advanced_multi_scale_second_attention_v2.py`
- **VFE**: `ImportanceGuidedMultiScaleVFE`
- **Features**: Learnable adaptive voxelization, multi-scale processing, Gumbel-Softmax scale selection
- **Output Channels**: 65 (64 features + 1 scale info)
- **Work Dir**: `./work_dirs/backbone_channel_fix`

### 2. Vanilla SECOND Baseline
**File**: `configs/vanilla_second_fair_benchmark.py`  
- **VFE**: `HardSimpleVFE` (standard)
- **Features**: Fixed voxelization, single-scale processing
- **Output Channels**: 4 (basic features only)
- **Work Dir**: `./work_dirs/vanilla_second_fair_benchmark`

## Identical Settings (Fair Comparison)

| Component | Setting | Value |
|-----------|---------|-------|
| **Data Root** | KITTI dataset path | `/home/daham/mmdetection_project/dataset/KITTI/` |
| **Point Cloud Range** | Spatial bounds | `[0, -40, -3, 70.4, 40, 1]` |
| **Voxel Size** | Base resolution | `[0.05, 0.05, 0.1]` |
| **Max Voxels** | Memory limits | `(12000, 30000)` |
| **Max Points per Voxel** | Point capacity | `5` |
| **Middle Encoder** | Architecture | `SparseEncoder` with 256 output channels |
| **Backbone** | Architecture | `SECOND` with 512 input channels |
| **Optimizer** | Type & Settings | `AdamW`, lr=0.001, weight_decay=0.05 |
| **Scheduler** | Learning rate | `LinearLR` + `CosineAnnealingLR` |
| **Batch Size** | Training | `1` |
| **Epochs** | Training duration | `1` (for quick testing) |
| **Dataloader** | Workers | `num_workers=2`, `persistent_workers=True` |

## Key Differences (Controlled Variables)

| Aspect | Adaptive VFE | Vanilla VFE | Impact |
|--------|-------------|-------------|---------|
| **Voxelization** | Learnable adaptive sizes | Fixed uniform size | Core research contribution |
| **Scale Selection** | Gumbel-Softmax (differentiable) | None | Enable end-to-end learning |
| **Multi-Scale Processing** | 3 scales (0.02m, 0.15m, 0.6m) | Single scale (0.05m) | Information-adaptive resolution |
| **Feature Channels** | 65 (64 + scale info) | 4 (basic only) | Richer feature representation |
| **Middle Encoder Input** | 65 channels | 4 channels | Different capacity requirements |

## Expected Performance Differences

### Adaptive VFE Advantages:
1. **Better Detail Preservation**: Fine scales (0.02m) for dense regions
2. **Computational Efficiency**: Coarse scales (0.6m) for sparse regions  
3. **Learned Optimization**: End-to-end trainable voxel sizes
4. **Information Adaptation**: Voxel resolution matches local information density

### Vanilla VFE Characteristics:
1. **Uniform Processing**: Same resolution everywhere
2. **Simpler Pipeline**: No scale selection overhead
3. **Fixed Memory**: Predictable memory usage
4. **Standard Baseline**: Well-established performance reference

## Benchmarking Commands

### Train Adaptive VFE:
```bash
python tools/train.py configs/advanced_multi_scale_second_attention_v2.py
```

### Train Vanilla Baseline:
```bash
python tools/train.py configs/vanilla_second_fair_benchmark.py
```

### Compare Results:
```bash
# View adaptive results
tensorboard --logdir=./work_dirs/backbone_channel_fix

# View vanilla results  
tensorboard --logdir=./work_dirs/vanilla_second_fair_benchmark
```

## Metrics to Compare

1. **Detection Accuracy**: mAP, precision, recall
2. **Training Speed**: Iterations per second
3. **Memory Usage**: GPU memory consumption
4. **Convergence**: Loss curves and training stability
5. **Inference Speed**: FPS during evaluation

## PhD Research Validation

This fair comparison will validate:
- ✅ **Adaptive voxelization improves accuracy** vs fixed voxelization
- ✅ **End-to-end learning** of optimal voxel sizes
- ✅ **Information-based adaptation** provides performance gains
- ✅ **Multi-scale processing** maintains computational efficiency

The identical settings ensure any performance differences are directly attributable to the adaptive voxelization innovation.
