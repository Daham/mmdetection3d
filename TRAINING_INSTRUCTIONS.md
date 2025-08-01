# Training Instructions for Remote Machine

## Issue Resolution Summary

The import error was caused by the multi-resolution encoder trying to import `spconv` even when not being used. I've fixed this by:

1. **Made spconv import conditional** in `multi_resolution_sparse_encoder.py`
2. **Made module registration conditional** in `middle_encoders/__init__.py`
3. **Created fallback configs** that don't require spconv

## Training Options (Ranked by Simplicity)

### Option 1: Use Your Existing AdaptiveVFE (SAFEST)
```bash
python tools/train.py configs/second/existing_adaptive_minimal.py \
    --work-dir work_dirs/existing_adaptive \
    --auto-scale-lr
```
**Benefits:**
- ✅ Uses your existing working AdaptiveVFE
- ✅ No new dependencies required
- ✅ Minimal changes from base SECOND

### Option 2: New Enhanced AdaptiveVFE (NO SPCONV NEEDED)
```bash
python tools/train.py configs/second/absolute_minimal_adaptive.py \
    --work-dir work_dirs/absolute_minimal \
    --auto-scale-lr
```
**Benefits:**
- ✅ Tests new enhanced adaptive VFE
- ✅ No spconv dependency
- ✅ Single scale adaptive voxelization

### Option 3: Multi-Resolution (REQUIRES SPCONV)
First install spconv:
```bash
pip install spconv-cu118  # or spconv-cu116/cu117 based on your CUDA version
```

Then run:
```bash
python tools/train.py configs/second/ultra_minimal_multi_res.py \
    --work-dir work_dirs/ultra_minimal \
    --auto-scale-lr
```

## Debugging Commands

### 1. Test Config Loading
```bash
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/second/existing_adaptive_minimal.py')
print('✅ Config loads successfully')
print(f'VFE type: {cfg.model.voxel_encoder.type}')
"
```

### 2. Test Module Import
```bash
python -c "
from mmdet3d.models.voxel_encoders.adaptive_vfe import AdaptiveVFE
print('✅ AdaptiveVFE imports successfully')
"
```

### 3. Check spconv availability
```bash
python -c "
try:
    import spconv
    print(f'✅ spconv available: {spconv.__version__}')
except ImportError:
    print('❌ spconv not available')
"
```

## File Status After Fixes

### Fixed Files:
- ✅ `mmdet3d/models/middle_encoders/multi_resolution_sparse_encoder.py` - Optional spconv import
- ✅ `mmdet3d/models/middle_encoders/__init__.py` - Conditional registration

### New Minimal Configs:
- ✅ `configs/second/existing_adaptive_minimal.py` - Uses your working AdaptiveVFE
- ✅ `configs/second/absolute_minimal_adaptive.py` - Uses new EnhancedAdaptiveVFE (no spconv)
- ✅ `configs/second/ultra_minimal_multi_res.py` - Multi-resolution (needs spconv)

## Expected Results

### Option 1 (Existing AdaptiveVFE):
- Should work immediately since you've used this before
- Tests basic adaptive voxelization concept
- Baseline for comparison

### Option 2 (Enhanced AdaptiveVFE):
- Tests improved adaptive VFE with richer features
- No multi-resolution complexity
- Good stepping stone

### Option 3 (Multi-Resolution):
- Tests full multi-resolution adaptive system
- Requires spconv installation
- Complete research implementation

## Troubleshooting

### If you get import errors:
1. Check that you're in the MMDetection3D directory
2. Ensure PYTHONPATH includes the project directory
3. Verify the conda environment is activated

### If spconv is needed but not available:
```bash
# Check CUDA version
nvcc --version

# Install appropriate spconv version
pip install spconv-cu118  # For CUDA 11.8
pip install spconv-cu117  # For CUDA 11.7
pip install spconv-cu116  # For CUDA 11.6
```

### If training fails:
1. Start with Option 1 (existing_adaptive_minimal.py)
2. Check GPU memory usage
3. Reduce batch size if needed
4. Check dataset path is correct

## Recommendation

**Start with Option 1** (`existing_adaptive_minimal.py`) since it uses your already-working AdaptiveVFE. Once that works, you can progress to the enhanced versions.

The fixes I made should resolve the import issues you encountered!
