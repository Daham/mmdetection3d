# Alternative Configuration: sparse_unet
# Alternative 3: SparseUNet
# Use this if you want skip connections and multi-scale features
middle_encoder=dict(
    type='SparseUNet',
    in_channels=64,
    sparse_shape=[41, 1600, 1408],
    base_channels=16,
    output_channels=128),
