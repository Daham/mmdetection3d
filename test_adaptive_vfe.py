from mmdet3d.registry import MODELS

# Test if AdaptiveVFE is registered
try:
    adaptive_vfe_cfg = dict(
        type='AdaptiveVFE',
        base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
        embed_dims=256,
        num_heads=8,
        num_layers=3,
        pos_encoding_cfg=dict(input_channel=3, num_pos_feats=256),
        attention_threshold=0.5,
        voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]
    )

    model = MODELS.build(adaptive_vfe_cfg)
    print("✅ AdaptiveVFE successfully registered and can be built!")
    print(f"Model type: {type(model)}")

except Exception as e:
    print(f"❌ Error: {e}")
