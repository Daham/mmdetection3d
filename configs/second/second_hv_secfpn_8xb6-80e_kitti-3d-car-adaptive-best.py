# configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car-adaptive-best.py
# OPTIMAL CONFIG FOR ADAPTIVE VOXELIZATION
# Using AdaptiveSparseEncoderV3Simple as the best balance of performance and adaptivity

# Custom imports to ensure adaptive modules are loaded
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.adaptive_vfe',
             'mmdet3d.models.middle_encoders.adaptive_sparse_encoder_v3'],
    allow_failed_imports=False)

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

model = dict(
    # Replace voxel encoder with adaptive version
    voxel_encoder=dict(
        type='AdaptiveVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='density',
        num_scales=3),
    
    # Replace middle encoder with adaptive version
    middle_encoder=dict(
        type='AdaptiveSparseEncoderV3Simple',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        adaptive_channel_boost=64))

# Override training epochs to 2
train_cfg = dict(max_epochs=2, val_interval=1)

# Optimizer with adaptive-friendly settings
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.01))
