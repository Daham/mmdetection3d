# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_scatter import PointPillarsScatter
from .sparse_encoder import SparseEncoder, SparseEncoderSASSD
# from .adaptive_sparse_encoder import AdaptiveSparseEncoder  # Disabled due to spconv dependency
from .adaptive_sparse_encoder_v3 import AdaptiveSparseEncoderV3, AdaptiveSparseEncoderV3Simple

# Conditional import for multi-resolution encoder (requires spconv)
try:
    from .multi_resolution_sparse_encoder import MultiResolutionSparseEncoder
    _multi_res_available = True
except ImportError:
    _multi_res_available = False

from .sparse_unet import SparseUNet
from .voxel_set_abstraction import VoxelSetAbstraction

__all__ = [
    'PointPillarsScatter', 'SparseEncoder', 'SparseEncoderSASSD',  # 'AdaptiveSparseEncoder',
    'AdaptiveSparseEncoderV3', 'AdaptiveSparseEncoderV3Simple', 
    'SparseUNet', 'VoxelSetAbstraction'
]

# Add MultiResolutionSparseEncoder to __all__ only if spconv is available
if _multi_res_available:
    __all__.append('MultiResolutionSparseEncoder')
