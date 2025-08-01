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
    '../_base_/datasets/kitti-3d-car.py',  # Same as your working config
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

voxel_size = [0.5, 0.5, 0.5]  # MATCH vanilla SECOND
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

model = dict(
    # UPDATE: Match voxel_size with vanilla
    data_preprocessor=dict(
        voxel_layer=dict(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    
    # Adaptive voxel encoder - but match vanilla output channels
    voxel_encoder=dict(
        type='AdaptiveVFE',
        base_vfe_cfg=dict(
            type='HardSimpleVFE',
            num_features=4  # Output 4 channels like vanilla
        )),
    
    # Adaptive middle encoder with corrected sparse_shape for voxel_size [0.5, 0.5, 0.5]
    middle_encoder=dict(
        type='AdaptiveSparseEncoderV3Simple',
        in_channels=4,  # Match vanilla input channels
        sparse_shape=[8, 160, 141],  # Correct for voxel_size [0.5, 0.5, 0.5]
        order=('conv', 'norm', 'act')),
    
    # Match vanilla bbox configuration exactly
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False))

# Override training epochs to match vanilla
train_cfg = dict(max_epochs=2, val_interval=1)

# MATCH vanilla optimizer exactly
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01))
