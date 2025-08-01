# Alternative Configuration: full_adaptive
# Alternative 1: Full Adaptive Features
# Use this for research and maximum adaptive capability
middle_encoder=dict(
    type='AdaptiveSparseEncoderV3',
    in_channels=64,
    sparse_shape=[41, 1600, 1408],
    base_channels=16,
    output_channels=128,
    adaptive_processing=True,
    adaptive_attention=True,
    multi_scale_fusion=True,
    content_aware_weighting=True,
    adaptive_feature_dim=32),
