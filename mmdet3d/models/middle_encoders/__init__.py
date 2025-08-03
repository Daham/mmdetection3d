# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_scatter import PointPillarsScatter
from .sparse_encoder import SparseEncoder, SparseEncoderSASSD
from .sparse_unet import SparseUNet
from .voxel_set_abstraction import VoxelSetAbstraction
from .multi_scale_parallel_middle_encoder import MultiScaleParallelMiddleEncoder
# 🚀 Optimized adaptive middle encoder (temporarily disabled due to spconv issues)
# from .optimized_multi_scale_parallel_middle_encoder import OptimizedMultiScaleParallelMiddleEncoder
from .fallback_multi_scale_parallel_middle_encoder import FallbackMultiScaleParallelMiddleEncoder
from .efficient_multi_scale_parallel_middle_encoder import EfficientMultiScaleParallelMiddleEncoder

__all__ = [
    'PointPillarsScatter', 'SparseEncoder', 'SparseEncoderSASSD',
    'SparseUNet', 'VoxelSetAbstraction', 'MultiScaleParallelMiddleEncoder',
    # 'OptimizedMultiScaleParallelMiddleEncoder'  # Temporarily disabled
    'FallbackMultiScaleParallelMiddleEncoder',  # Spconv-free fallback solution
    'EfficientMultiScaleParallelMiddleEncoder',  # High-performance vectorized solution
]
