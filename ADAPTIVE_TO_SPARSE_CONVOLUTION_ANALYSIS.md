# Data Flow Analysis: Adaptive Voxelization to Sparse Convolution

This document analyzes how data flows from adaptive voxelization through to sparse convolution in the MMDetection3D pipeline, highlighting the key interfaces and data transformations.

## Pipeline Overview

```
Raw Points → Adaptive Voxelization → Bridge/Mapping → Sparse Convolution → Features
    [N, C]         [M, feat_dim]        [K, feat_dim]      [K, feat_dim]     [output]
                      (irregular)         (regular)          (regular)
```

## 1. Input Data Format

### Raw Point Cloud
- **Format**: `[N, C]` where N = number of points, C = channels (typically 4: x, y, z, intensity)
- **Example**: `[100000, 4]` for a typical KITTI frame
- **Range**: Bounded by `point_cloud_range` parameter

## 2. Adaptive Voxelization Output

### True Adaptive Voxelizer Output
```python
# TrueAdaptiveVoxelizer.forward() returns:
voxel_features: torch.Tensor  # [M, feat_dim] - processed voxel features
voxel_coords: torch.Tensor    # [M, 4] - (batch_idx, z, y, x) coordinates  
adaptive_info: Dict           # metadata about adaptation
```

**Key Issue**: `voxel_coords` may have irregular spacing due to adaptive voxel sizes:
- Voxel at coord [0, 10, 20, 30] might represent a 0.05m voxel
- Voxel at coord [0, 10, 20, 31] might represent a 0.1m voxel
- This breaks sparse convolution's assumption of regular grid structure

## 3. Bridge Mapping Solution

### AdaptiveToRegularBridge Transformation
The bridge solves the compatibility problem by:

1. **Adaptive Voxelization**: Creates variable-size voxels based on local conditions
2. **Regular Grid Mapping**: Maps adaptive voxels back to a regular base grid
3. **Conflict Resolution**: Handles multiple adaptive voxels mapping to same regular cell

```python
# AdaptiveToRegularBridge.forward() returns:
regular_features: torch.Tensor  # [K, feat_dim] - features on regular grid
regular_coords: torch.Tensor    # [K, 4] - regular grid coordinates
adaptive_info: Dict             # adaptation metadata
```

**Critical Properties**:
- `regular_coords` follow regular grid structure required by sparse convolution
- Coordinates are multiples of base voxel size
- No gaps or irregular spacing in the coordinate system

## 4. Sparse Convolution Interface

### Expected Input Format
Sparse convolution (spconv) expects:
```python
SparseConvTensor(
    features=features,     # [N, C] - feature vectors
    indices=coordinates,   # [N, 4] - (batch, z, y, x) in regular grid
    spatial_shape=shape,   # [D, H, W] - 3D grid dimensions
    batch_size=batch_size
)
```

### Key Requirements
1. **Regular Grid**: Coordinates must form a regular 3D grid
2. **Integer Coordinates**: All coordinates must be non-negative integers
3. **Consistent Spacing**: Coordinate differences represent actual spatial relationships
4. **Spatial Shape**: Must match the maximum coordinate values

## 5. Data Flow Examples

### Example 1: Feature-Level Adaptation (EnhancedAdaptiveVFE)
```
Points [100000, 4] → Voxelization [5000, 32, 4] → EnhancedAdaptiveVFE [5000, 64] → SparseConv
                                                       ↑
                                              Adaptive features but
                                              regular grid structure
```
- **Grid Structure**: Unchanged (regular)
- **Adaptation**: Feature-level only
- **Sparse Conv Compatibility**: ✅ Perfect

### Example 2: True Adaptive with Bridge (AdaptiveToRegularBridge)
```
Points [100000, 4] → Adaptive Voxelization [3000, 64] → Bridge Mapping [4500, 64] → SparseConv
                           ↑ (irregular)                      ↑ (regular)
                      Variable voxel sizes              Regular grid structure
```
- **Grid Structure**: Irregular → Regular (via bridge)
- **Adaptation**: True voxel size adaptation
- **Sparse Conv Compatibility**: ✅ Via bridge mapping

### Example 3: Multi-Resolution (MultiResolutionSparseEncoder)
```
Points [100000, 4] → Split by Resolution → Multiple Sparse Grids → Fusion → Output
                           ↓
              Level 0: [2000, 64] (fine)
              Level 1: [1500, 64] (medium)  
              Level 2: [1000, 64] (coarse)
```
- **Grid Structure**: Multiple regular grids
- **Adaptation**: Multi-scale processing
- **Sparse Conv Compatibility**: ✅ Each level is regular

## 6. Coordinate System Analysis

### Regular Grid Coordinates (Compatible)
```python
# Base voxel size: [0.05, 0.05, 0.1]
# Point at (1.0, 2.0, 0.5) → Voxel coord (20, 40, 5)
# Point at (1.05, 2.05, 0.6) → Voxel coord (21, 41, 6)
# Regular spacing: each coordinate step = base voxel size
```

### Adaptive Grid Coordinates (Incompatible)
```python
# Adaptive voxel sizes: [0.025, 0.025, 0.05] to [0.2, 0.2, 0.4]
# Point at (1.0, 2.0, 0.5) with small voxel → Coord could be (40, 80, 10)
# Point at (2.0, 3.0, 0.9) with large voxel → Coord could be (10, 15, 2)
# Irregular spacing: coordinate steps vary with local voxel size
```

## 7. Bridge Mapping Algorithm

### Core Mapping Strategy
1. **Adaptive Voxelization**: Use local density/distance to determine optimal voxel sizes
2. **Coordinate Transformation**: Map adaptive voxel centers to regular grid coordinates
3. **Conflict Resolution**: When multiple adaptive voxels map to same regular cell:
   - `'max'`: Take maximum feature values
   - `'average'`: Average feature values
   - `'weighted_average'`: Learned weighted combination

```python
# Mapping example:
adaptive_coord = compute_adaptive_coordinate(point, adaptive_voxel_size)
regular_coord = map_to_regular_grid(point, base_voxel_size)

# Multiple adaptive voxels may map to same regular coordinate
regular_grid[regular_coord] = resolve_conflict(
    existing_features, new_features, method='weighted_average'
)
```

## 8. Performance Implications

### Memory Usage
- **Feature-Level Adaptation**: Same memory as standard voxelization
- **True Adaptive**: Potentially higher memory due to bridge mapping
- **Multi-Resolution**: Higher memory (multiple grids)

### Computational Cost
- **Feature-Level**: Low overhead (just feature processing)
- **True Adaptive**: Moderate overhead (adaptation + mapping)
- **Multi-Resolution**: High overhead (multiple sparse convolutions)

### Quality vs. Efficiency Trade-off
1. **EnhancedAdaptiveVFE**: Fast, good for subtle adaptations
2. **AdaptiveToRegularBridge**: Moderate cost, true adaptive voxelization
3. **MultiResolutionSparseEncoder**: Expensive, highest quality

## 9. Integration Points

### Config File Integration
```python
model = dict(
    voxel_encoder=dict(
        type='AdaptiveToRegularBridge',  # Adaptive voxelization with bridge
        # ... parameters
    ),
    middle_encoder=dict(
        type='SparseEncoder',  # Standard sparse convolution
        # ... parameters  
    )
)
```

### Data Pipeline Integration
```python
# In data pipeline:
points → voxelization → adaptive_vfe → sparse_encoder → neck → head

# The bridge ensures voxelization output is compatible with sparse_encoder
```

## 10. Testing and Validation

### Coordinate Validation
```python
def validate_sparse_compatibility(coords, spatial_shape):
    """Validate coordinates are compatible with sparse convolution."""
    # Check coordinate bounds
    assert coords.min() >= 0
    assert coords[:, 1].max() < spatial_shape[0]  # Z
    assert coords[:, 2].max() < spatial_shape[1]  # Y  
    assert coords[:, 3].max() < spatial_shape[2]  # X
    
    # Check for regular spacing (for true regular grids)
    # This may not hold for bridged adaptive grids
```

### Feature Validation
```python
def validate_feature_consistency(features, coords):
    """Validate feature and coordinate tensors are consistent."""
    assert features.shape[0] == coords.shape[0]
    assert coords.shape[1] == 4  # (batch, z, y, x)
    assert not torch.isnan(features).any()
    assert not torch.isnan(coords).any()
```

## Summary

The adaptive voxelization to sparse convolution pipeline requires careful handling of coordinate systems:

1. **Feature-level adaptation** (EnhancedAdaptiveVFE) maintains regular grids and is fully compatible
2. **True adaptive voxelization** (TrueAdaptiveVoxelizer) creates irregular grids that need bridge mapping
3. **Bridge mapping** (AdaptiveToRegularBridge) solves compatibility by mapping irregular to regular grids
4. **Multi-resolution processing** maintains compatibility by using multiple regular grids

The key insight is that sparse convolution fundamentally requires regular grid structure, so any adaptive voxelization must either:
- Adapt only at the feature level (keeping grid regular)
- Map irregular adaptive grids back to regular structure
- Use multiple regular grids at different resolutions

All implemented solutions maintain this compatibility while enabling various forms of adaptive processing.
