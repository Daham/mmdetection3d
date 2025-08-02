# Performance Optimization Guide

## Current Performance Issues Identified

### Timing Analysis
- **Per iteration**: 3.05 seconds
- **Per epoch**: ~1.5-2 hours (1667 iterations)
- **Data loading**: 0.1133s (good)
- **Compute time**: ~2.9s per iteration (too slow)

### Memory Usage
- **4766 MB per iteration** - high memory consumption
- **Voxel processing**: 95K→19K voxels per batch (heavy pruning)

## Immediate Optimizations

### 1. Reduce Batch Complexity
```bash
# Edit config to use smaller voxel grid
configs/second/adaptive_voxel_*.py:
voxel_size = [0.2, 0.2, 0.4]  # Increase from [0.05, 0.05, 0.1]
point_cloud_range = [-50, -50, -5, 50, 50, 3]  # Reduce range
```

### 2. Increase Batch Size (if GPU memory allows)
```python
# In training config:
train_dataloader = dict(
    batch_size=4,  # Increase from 1-2
    num_workers=4,
    persistent_workers=True
)
```

### 3. Use Mixed Precision Training
```python
# Add to config:
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Automatic Mixed Precision
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01)
)
```

### 4. Enable CuDNN Benchmark
```python
# In config:
env_cfg = dict(
    cudnn_benchmark=True,  # Speed up training
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)
```

## Quick Test Commands

### Fast Training Test (5 iterations only)
```bash
cd /home/daham/mmdetection_project/mmdetection3d
/home/daham/mmdetection_project/mmdet_env/bin/python tools/train.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
    --cfg-options train_cfg.max_iters=5 \
    --work-dir work_dirs/quick_test
```

### Profile Memory Usage
```bash
/home/daham/mmdetection_project/mmdet_env/bin/python -c "
import torch
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
print(f'GPU Memory Available: {torch.cuda.memory_reserved(0)/1e9:.1f}GB')
"
```

## Expected Performance After Optimization
- **Target**: <1s per iteration
- **Per epoch**: <30 minutes
- **Memory**: <3GB per iteration

## Commands to Test Optimizations

1. **Quick performance test**:
```bash
/home/daham/mmdetection_project/mmdet_env/bin/python tools/train.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
    --cfg-options train_cfg.max_iters=10
```

2. **Monitor GPU usage**:
```bash
watch -n 1 nvidia-smi
```

3. **Check data loading speed**:
```bash
/home/daham/mmdetection_project/mmdet_env/bin/python tools/analysis_tools/benchmark_data_loading.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py
```
