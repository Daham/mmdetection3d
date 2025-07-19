_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]
# point_cloud_range=[0, -25, -1.5, 40, 25, 1.5]
voxel_size = [0.5, 0.5, 0.5]
point_cloud_range=[0, -40, -3, 70.4, 40, 1]

# FIX 3: Recalculate the sparse_shape for the middle_encoder based on the new voxel_size.
# The original shape [41, 1600, 1408] was based on a 0.05m voxel size.
# This new shape [9, 160, 141] matches the 0.5m voxel size and prevents memory errors.
# The order is [z, y, x].
sparse_shape = [
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]) + 1,
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
]

model = dict(
    # FIX 1: Override the data_preprocessor to use the new voxel_size.
    # This ensures voxels are CREATED and INTERPRETED with the same dimensions.
    data_preprocessor=dict(
        voxel_layer=dict(
            voxel_size=voxel_size
        )
    ),
    # Configure the custom AdaptiveVFE module as the new voxel_encoder.
    # voxel_encoder=dict(
    #     type='AdaptiveVFE',
    #     base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
    #     embed_dims=256,
    #     num_heads=8,
    #     num_layers=3,
    #     pos_encoding_cfg=dict(type='ConvBNPositionalEncoding', input_channel=3, num_pos_feats=256),
    #     attention_threshold=0.5,
    #     voxel_size=voxel_size,
    #     point_cloud_range=point_cloud_range),

    # Override the middle_encoder with the new, correct sparse_shape.
    middle_encoder=dict(
        sparse_shape=sparse_shape
    ),

    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            # ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
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
