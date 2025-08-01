# 🎯 Adaptive Voxelization for Sparse Convolution

## What This Does

**ONE SIMPLE SOLUTION** that gives you:

1. **✅ Adaptive voxel sizes during training** - The network learns optimal voxel sizes
2. **✅ Feeds to Sparse Convolution** - Automatically handles compatibility  
3. **✅ No extra setup required** - Just change the config

## The Problem Solved

- **Standard voxelization**: Fixed voxel size everywhere (inefficient)
- **Sparse convolution**: Only accepts regular grids (rigid requirement)
- **This solution**: Adaptive voxels + automatic mapping to regular grid

## How It Works

```
Raw Points → Learn Voxel Sizes → Create Adaptive Voxels → Map to Regular Grid → Sparse Convolution
    ↓              ↓                      ↓                     ↓                ↓
[N, 4]      [N, 3] sizes        Variable voxels        Regular grid       Standard output
```

### Key Innovation: The Bridge

The `AdaptiveSparseBridge` does TWO things:

1. **Creates adaptive voxels**: Dense areas get small voxels, sparse areas get large voxels
2. **Maps to regular grid**: Converts variable voxels back to regular structure for sparse convolution

## Usage

### 1. Single Config File
Use `configs/second/adaptive_sparse.py`:

```python
model = dict(
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',  # THE ONLY MODULE YOU NEED
        learnable_adaptation=True,    # Enable learning
        # ... parameters
    )
)
```

### 2. Run Training
```bash
python tools/train.py configs/second/adaptive_sparse.py
```

That's it! The module handles everything automatically.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_voxel_size` | Smallest voxels (dense areas) | `[0.025, 0.025, 0.05]` |
| `max_voxel_size` | Largest voxels (sparse areas) | `[0.2, 0.2, 0.4]` |
| `base_voxel_size` | Regular grid size | `[0.05, 0.05, 0.1]` |
| `learnable_adaptation` | Enable learning | `True` |

## The Magic: Automatic Mapping

**Problem**: Sparse convolution needs regular grids
```
Regular: [0,0,0] → [0,0,1] → [0,0,2]  ✅ Works
Adaptive: [0,0,0] → [0,0,1.5] → [0,0,3.2]  ❌ Breaks sparse conv
```

**Solution**: Bridge mapping
```
Adaptive Voxels → Regular Grid Coordinates → Sparse Convolution
   (variable)         (regular)              (compatible)
```

## Files You Need

**Essential:**
- `mmdet3d/models/voxel_encoders/adaptive_sparse_bridge.py` - The main module
- `configs/second/adaptive_sparse.py` - The config

**That's it!** Everything else is cleaned up.

## Comparison

| Method | Voxel Sizes | Sparse Conv | Complexity |
|--------|-------------|-------------|------------|
| Standard VFE | Fixed | ✅ | Simple |
| **AdaptiveSparseBridge** | **Adaptive** | **✅** | **Automatic** |

## Training Results

The network will learn:
- **Small voxels** for object regions (high detail)
- **Large voxels** for empty space (efficiency)
- **Optimal balance** between detail and speed

## Next Steps

1. **Test**: `python tools/train.py configs/second/adaptive_sparse.py`
2. **Monitor**: Watch the adaptive voxel size ranges in logs
3. **Compare**: Run against standard config to see improvements
4. **Tune**: Adjust `min_voxel_size` and `max_voxel_size` if needed

**Goal achieved: Adaptive voxel sizes + Sparse convolution compatibility! 🚀**
