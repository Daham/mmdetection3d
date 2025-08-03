# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_encoder import DynamicPillarFeatureNet, PillarFeatureNet
from .voxel_encoder import (DynamicSimpleVFE, DynamicVFE, HardSimpleVFE,
                            HardVFE, SegVFE)

# Research: Multi-scale adaptive voxelization 
from .multi_scale_adaptive_voxel import (MultiScaleAdaptiveVoxelEncoder, 
                                        MultiScaleImportancePredictor,
                                        MultiScaleFeatureFusion)

# 🚀 Optimized adaptive voxelization
from .optimized_multi_scale_adaptive_voxel import OptimizedMultiScaleAdaptiveVoxelEncoder

__all__ = [
    'DynamicPillarFeatureNet', 'PillarFeatureNet', 'DynamicSimpleVFE', 'DynamicVFE', 
    'HardSimpleVFE', 'HardVFE', 'SegVFE',
    'MultiScaleAdaptiveVoxelEncoder', 'MultiScaleImportancePredictor', 'MultiScaleFeatureFusion',
    'OptimizedMultiScaleAdaptiveVoxelEncoder'
]
