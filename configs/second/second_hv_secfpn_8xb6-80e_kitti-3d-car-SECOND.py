_base_ = './second_hv_secfpn_8xb6-80e_kitti-3d-car-SECOND.py'

# Import FocalLoss from mmdet if not already registered
custom_imports = dict(
    imports=['mmdet.models.losses.focal_loss'],  # or use your custom path if different
    allow_failed_imports=False
)

# Only override the voxel encoder to use LearnableVFE
model = dict(
    voxel_encoder=dict(
        type='LearnableVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]
    ),
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',  # Make sure FocalLoss is imported and registered
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        )
        # Keep other loss_* and bbox_coder settings from base config unless you want to override
    )
)

# Keep your optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
)
