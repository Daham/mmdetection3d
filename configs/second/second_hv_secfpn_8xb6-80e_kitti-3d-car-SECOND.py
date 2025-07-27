_base_ = './second_hv_secfpn_8xb6-80e_kitti-3d-car-SECOND.py'

# Only override the voxel encoder to use LearnableVFE
model = dict(
    pts_voxel_encoder=dict(
        type='LearnableVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    # Keep everything else from the base config
)

# Keep your optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
)