_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]


voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/' 

model = dict(
    # Use standard VoxelNet detector for better compatibility
    # LearnableVFE with region-adaptive feature weighting
    voxel_encoder=dict(
        type='LearnableVFE',
        in_channels=4,  # x, y, z, intensity
        feat_channels=[64],  # Output 64 channels
        with_distance=True,  # Enable region-adaptive distance features
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    # Standard sparse encoder (back to safe configuration)
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,  # Must match LearnableVFE output
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
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
        max_epochs=5,
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

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
)