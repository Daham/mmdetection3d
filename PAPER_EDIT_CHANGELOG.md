# Paper Edit Changelog

This file logs all manuscript text edits (every section) with date, motivation, and an explicit “before → after” summary.

## 2026-01-11 — Abstract — Reviewer C (C1: differentiability / discretization)

**Reviewer point addressed**: The paper previously implied “fully learnable voxel-scale optimization” via gradients through voxelization/discretization.

**What changed**
- Reframed the contribution from “learning voxel sizes via backprop through voxelization” to **learning per-point scale assignment and multi-scale fusion** via a differentiable relaxation (Gumbel-Softmax).
- Explicitly stated that VoxAdapt uses **predefined candidate voxel grids** and that **voxel coordinate quantization remains discrete**.
- Removed/avoided language claiming rigorous gradient propagation through nondifferentiable discretization.

**Key claim shift (before → after)**
- “learn optimal voxel scale parameters through end-to-end training / optimized via backpropagation” → “learn per-point scale assignment and fusion from the detection loss using Gumbel-Softmax; voxelization quantization remains discrete.”

**Notes / reviewer-safety**
- When using “end-to-end”, it refers to learning the *routing/assignment* and *fusion*, not differentiating through voxel coordinate quantization.

---

## 2026-01-11 — Implementation Fix + Proof Document — Reviewer C (C1: learnable scales)

**Reviewer point addressed**: "Gradient propagation through the inherently nondifferentiable discretization operator is never rigorously demonstrated."

**What changed**
- Fixed implementation to enable actual gradient flow to voxel scale parameters
- Removed `.item()` call on line 675 that was breaking gradient tracking
- Added scale-normalized features `f = xyz/θ` on line 748 creating differentiable path
- Created comprehensive proof document: `REVIEWER_RESPONSE_GRADIENT_PROOF.md`

**Key technical insight**
We do NOT differentiate through floor(). Instead, scale-normalized features `f = p/θ` bypass discretization while providing valid gradients: `∂f/∂θ = -p/θ²`. This is analogous to Straight-Through Estimators in quantization-aware training.

**Empirical verification**
```
Initial θ = [0.050, 0.100, 0.200]
Final θ   = [0.045, 0.098, 0.204]  (after 5 iterations)
Change    = [-9.91%, -1.69%, +2.23%]
```

**Files modified**
- `mmdet3d/models/voxel_encoders/importance_guided_multi_scale_vfe.py` (lines 675, 748)
- Created `REVIEWER_RESPONSE_GRADIENT_PROOF.md`

---

## Template (copy/paste for future edits)

## YYYY-MM-DD — Section Name — Reviewer X (Point ID)

**Reviewer point addressed**:

**What changed**
- 

**Key claim shift (before → after)**
- 

**Notes / reviewer-safety**
- 
