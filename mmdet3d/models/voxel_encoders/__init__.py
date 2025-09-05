# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_encoder import DynamicPillarFeatureNet, PillarFeatureNet
from .voxel_encoder import (DynamicSimpleVFE, DynamicVFE, HardSimpleVFE,
                            HardVFE, SegVFE)

# ✨ PhD Research: Importance-guided multi-scale VFE
from .importance_guided_multi_scale_vfe import (ImportanceGuidedMultiScaleVFE,
                                                 ScaleNet,
                                                 MultiScaleVoxelizer,
                                                 ScaleSpecificVFE,
                                                 RefactoredMultiScaleFeatureFusion,
                                                 LightweightPointImportanceNet)

# 🎯 Baseline: Fixed multi-scale VFE for comparison
from .fixed_multi_scale_vfe import (FixedMultiScaleVFE,
                                     FixedScaleVFE,
                                     FixedMultiScaleVoxelizer,
                                     FixedMultiScaleFeatureFusion)

# 🆕 Fixed Multi-Scale VFE (clean baseline implementation)
from .fixed_multiscale_vfe import SimpleFixedMultiScaleVFE

__all__ = [
    'DynamicPillarFeatureNet', 'PillarFeatureNet', 'DynamicSimpleVFE', 'DynamicVFE', 
    'HardSimpleVFE', 'HardVFE', 'SegVFE',
    'ImportanceGuidedMultiScaleVFE', 'ScaleNet', 'MultiScaleVoxelizer', 
    'ScaleSpecificVFE', 'RefactoredMultiScaleFeatureFusion', 'LightweightPointImportanceNet',
    'FixedMultiScaleVFE', 'FixedScaleVFE', 'FixedMultiScaleVoxelizer', 'FixedMultiScaleFeatureFusion',
    'SimpleFixedMultiScaleVFE'  # Clean fixed multi-scale VFE implementation
]
