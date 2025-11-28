"""
Simple Baseline_03: Your Adaptive Multi-Scale Method (2 epochs)
Based on working SECOND config, just swap the voxel encoder
"""

_base_ = ['./second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py']

# Override with your adaptive voxel encoder
model = dict(
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        num_scales=3,
        voxel_scales=[0.05, 0.1, 0.2],
        vfe_channels=[32, 64],
        fusion_channels=64,
        output_channels=3,
        gumbel_temperature=2.0,
        max_num_points=5,
        max_voxels=(16000, 40000),
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        continuous_mode=False
    )
)

# Reduce LR for stability
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.01, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale=4096.0
)

# Smaller batch size for multi-scale
train_dataloader = dict(batch_size=4)

# Add EMA for stability
custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, priority='ABOVE_NORMAL', strict_load=False)
]
