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

# 🎯 Advanced multi-scale VFE with attention
from .multi_scale_vfe_with_attention import MultiScaleVFEWithAttention

# ✨ Importance-guided multi-scale VFE with point filtering
from .importance_guided_multi_scale_vfe import (ImportanceGuidedMultiScaleVFE,
                                               LightweightPointImportanceNet,
                                               ScaleSpecificLightweightVFE)

__all__ = [
    'DynamicPillarFeatureNet', 'PillarFeatureNet', 'DynamicSimpleVFE', 'DynamicVFE', 
    'HardSimpleVFE', 'HardVFE', 'SegVFE',
    'MultiScaleAdaptiveVoxelEncoder', 'MultiScaleImportancePredictor', 'MultiScaleFeatureFusion',
    'OptimizedMultiScaleAdaptiveVoxelEncoder', 'MultiScaleVFEWithAttention',
    'ImportanceGuidedMultiScaleVFE', 'LightweightPointImportanceNet', 'ScaleSpecificLightweightVFE'
]
