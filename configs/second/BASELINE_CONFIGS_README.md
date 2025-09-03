# 🎯 Adaptive Voxelization Research - Baseline Configurations

This directory contains the three core baseline configurations for comparative research on adaptive voxelization approaches.

## 📋 **Experimental Baselines**

### 1. **Baseline 01: Single-Scale (Standard SECOND)**
**File:** `baseline_01_single_scale_hardvfe.py`
- **VFE Type:** `HardSimpleVFE` (Standard SECOND)
- **Purpose:** Vanilla SECOND baseline for comparison
- **Voxel Scale:** Single fixed scale [0.1, 0.1, 0.2]m
- **Training:** 2 epochs, lr=0.001
- **Status:** ✅ Ready to run

### 2. **Baseline 02: Fixed Multi-Scale + Gumbel-Softmax**
**File:** `baseline_02_fixed_multiscale_gumbel.py`
- **VFE Type:** `FixedMultiScaleVFE`
- **Purpose:** Multi-scale processing with learnable weighted fusion
- **Voxel Scales:** Fixed [0.05, 0.1, 0.2]m with Gumbel-Softmax fusion
- **Training:** 3 epochs, lr=0.0001 (memory-optimized)
- **Status:** ✅ **COMPLETE** - Results: 69.15%/60.71%/54.17% 3D AP

### 3. **Baseline 03: Adaptive Multi-Scale + Learnable Scales**
**File:** `baseline_03_adaptive_multiscale_learnable.py`
- **VFE Type:** `ImportanceGuidedMultiScaleVFE`
- **Purpose:** Adaptive scale selection with learnable voxel parameters
- **Voxel Scales:** **Learnable** [0.05, 0.1, 0.2]m (PhD contribution)
- **Training:** 2 epochs, lr=0.001
- **Status:** ✅ Ready to run

## 🔬 **Research Framework**

| Aspect | Baseline 01 | Baseline 02 | Baseline 03 |
|--------|-------------|-------------|-------------|
| **Architecture** | Standard VFE | Fixed Multi-Scale | Adaptive Multi-Scale |
| **Scale Strategy** | Single Fixed | Fixed Multi-Scale | **Learnable Multi-Scale** |
| **Fusion Method** | None | Gumbel-Softmax | Importance-Guided |
| **Memory Usage** | 1x (baseline) | 3x overhead | 2x overhead |
| **Research Question** | Baseline performance | Multi-scale benefits | Adaptive benefits |

## 🎯 **Usage Instructions**

### Training Commands:
```bash
# Baseline 01: Single-Scale
python tools/train.py configs/second/baseline_01_single_scale_hardvfe.py

# Baseline 02: Fixed Multi-Scale (Already Complete)
python tools/train.py configs/second/baseline_02_fixed_multiscale_gumbel.py

# Baseline 03: Adaptive Multi-Scale  
python tools/train.py configs/second/baseline_03_adaptive_multiscale_learnable.py
```

## 📊 **Expected Research Outcomes**

1. **Baseline 01 → 02**: Quantify benefits of multi-scale processing
2. **Baseline 02 → 03**: Quantify benefits of adaptive scale selection
3. **Scale Learning**: Analyze how voxel scales evolve during training
4. **Performance Analysis**: Compare accuracy vs computational cost

## 🔧 **Development Notes**

- All configurations use consistent training parameters for fair comparison
- Memory optimizations applied to enable training on available hardware
- Gradient checkpointing and reduced batch sizes used where necessary
- Each baseline isolates specific research contributions

---
**Author:** Daham Pathiraja  
**Date:** September 3, 2025  
**Branch:** feature/adaptive-voxelization-research
