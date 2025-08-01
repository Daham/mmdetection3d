# Minimal adaptive config - start from proven working base
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Custom imports for adaptive modules
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.adaptive_vfe',
             'mmdet3d.models.middle_encoders.adaptive_sparse_encoder_v3'],
    allow_failed_imports=False)

# ONLY override what's absolutely necessary
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Use the EXACT same voxel size as the base model (DON'T override!)
# This should use voxel_size = [0.05, 0.05, 0.1] and sparse_shape = [41, 1600, 1408]

model = dict(
    # ONLY replace the voxel encoder with your adaptive version
    voxel_encoder=dict(
        type='AdaptiveVFE',
        in_channels=4,
        feat_channels=[4],  # Output same 4 channels as HardSimpleVFE
        with_distance=False,
        voxel_size=(0.05, 0.05, 0.1),  # Use base model's voxel size
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],  # Use base model's sparse_shape
        adaptation_method='density',
        num_scales=3),
    
    # Configure for single class (Car only)
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6]],  # Use same Z range as base
            sizes=[[3.9, 1.6, 1.56]],  # Car size
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

# Override training settings
train_cfg = dict(max_epochs=2, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
)
