# Alternative Configuration: vanilla_tuned
# Alternative 2: Vanilla SECOND with Manual Optimization
# Use this as baseline comparison
middle_encoder=dict(
    type='SparseEncoder',
    in_channels=64,
    sparse_shape=[41, 1600, 1408],
    base_channels=16,
    output_channels=128,
    encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
    encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1))),
