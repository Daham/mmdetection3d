# # === second_hv_secfpn_8xb6-80e_kitti-3d-car.py (FINAL VERSION) ===

# _base_ = [
#     '../_base_/models/second_hv_secfpn_kitti.py',
#     '../_base_/datasets/kitti-3d-car.py', '../_base_/schedules/cyclic-2e.py',
#     '../_base_/default_runtime.py'
# ]

# # ----------------------------------------------------------------------
# # 1. DEFINE SHARED PARAMETERS
# # ----------------------------------------------------------------------
# voxel_size = [0.5, 0.5, 0.5]
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]
# data_root = '/home/daham/mmdetection_project/dataset/KITTI/' #<-- Make sure this path is correct!

# # ----------------------------------------------------------------------
# # 2. OVERRIDE THE DATA LOADER TO DISABLE THE SAMPLER
# # This is the definitive fix for the crash. It defines a simple pipeline
# # without the 'ObjectSample' / 'DataBaseSampler' augmentation.
# # ----------------------------------------------------------------------
# train_dataloader = dict(
#     dataset=dict(
#         dataset=dict(
#             data_root=data_root,
#             ann_file='kitti_infos_train.pkl',
#             pipeline=[
#                 dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
#                 dict(type='LoadAnnotations3D', with_bbox_d=True, with_label_3d=True),
#                 dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
#                 dict(
#                     type='GlobalRotScaleTrans',
#                     rot_range=[-0.78539816, 0.78539816],
#                     scale_ratio_range=[0.95, 1.05]),
#                 dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#                 dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#                 dict(type='PointShuffle'),
#                 dict(type='Pack3DDetInputs', keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
#             ]
#         )
#     )
# )

# # ----------------------------------------------------------------------
# # 3. CONFIGURE THE MODEL WITH ALL FIXES
# # ----------------------------------------------------------------------
# # Recalculate sparse_shape for the new voxel_size. Order is [z, y, x].
# sparse_shape = [
#     int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]) + 1,
#     int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
#     int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
# ]

# model = dict(
#     # Ensure data preprocessor uses the correct voxel size
#     data_preprocessor=dict(
#         voxel_layer=dict(voxel_size=voxel_size)
#     ),
#     # Use your custom Voxel Encoder
#     voxel_encoder=dict(
#         type='AdaptiveVFE',
#         base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
#         embed_dims=256,
#         num_heads=8,
#         num_layers=3,
#         pos_encoding_cfg=dict(type='ConvBNPositionalEncoding', input_channel=3, num_pos_feats=256),
#         attention_threshold=0.3, # Using the less strict threshold
#         significance_percentile=0.5, # Using the less strict threshold
#         voxel_size=voxel_size,
#         point_cloud_range=point_cloud_range),
#     # Update middle_encoder with the correct grid shape to prevent memory errors
#     middle_encoder=dict(
#         sparse_shape=sparse_shape
#     ),
#     # Your custom bbox_head for 1 class
#     bbox_head=dict(
#         num_classes=1,
#         anchor_generator=dict(
#             _delete_=True,
#             type='Anchor3DRangeGenerator',
#             ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
#             sizes=[[3.9, 1.6, 1.56]],
#             rotations=[0, 1.57],
#             reshape_out=True)),
#     # Your custom training settings
#     train_cfg=dict(
#         _delete_=True,
#         max_epochs=5,
#         assigner=dict(
#             type='Max3DIoUAssigner',
#             iou_calculator=dict(type='BboxOverlapsNearest3D'),
#             pos_iou_thr=0.6,
#             neg_iou_thr=0.45,
#             min_pos_iou=0.45,
#             ignore_iof_thr=-1),
#         allowed_border=0,
#         pos_weight=-1,
#         debug=False))

# # ----------------------------------------------------------------------
# # 4. SET OPTIMIZER
# # ----------------------------------------------------------------------
# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
# )




# === second_hv_secfpn_8xb6-80e_kitti-3d-car.py (FINAL VERSION) ===

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# ----------------------------------------------------------------------
# 1. DEFINE SHARED PARAMETERS
# ----------------------------------------------------------------------
voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/' #<-- Make sure this path is correct!

# ----------------------------------------------------------------------
# 2. OVERRIDE THE DATA LOADER TO DISABLE THE SAMPLER
# This is the definitive fix for the crash. It defines a simple pipeline
# without the 'ObjectSample' / 'DataBaseSampler' augmentation.
# ----------------------------------------------------------------------
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                dict(type='LoadAnnotations3D', with_bbox_d=True, with_label_3d=True),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='PointShuffle'),
                dict(type='Pack3DDetInputs', keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
            ]
        )
    )
)

# ----------------------------------------------------------------------
# 3. CONFIGURE THE MODEL WITH ALL FIXES
# ----------------------------------------------------------------------
# Recalculate sparse_shape for the new voxel_size. Order is [z, y, x].
sparse_shape = [
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]) + 1,
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
]

model = dict(
    # Ensure data preprocessor uses the correct voxel size
    data_preprocessor=dict(
        voxel_layer=dict(voxel_size=voxel_size)
    ),
    # Use your custom Voxel Encoder
    voxel_encoder=dict(
        type='AdaptiveVFE',
        base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
        embed_dims=256,
        num_heads=8,
        num_layers=3,
        pos_encoding_cfg=dict(type='ConvBNPositionalEncoding', input_channel=3, num_pos_feats=256),
        attention_threshold=0.3, # Using the less strict threshold
        significance_percentile=0.5, # Using the less strict threshold
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    # Update middle_encoder with the correct grid shape to prevent memory errors
    middle_encoder=dict(
        sparse_shape=sparse_shape
    ),
    # Your custom bbox_head for 1 class
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -3.0, 70.4, 40.0, 1.0]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # Your custom training settings
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

# ----------------------------------------------------------------------
# 4. SET OPTIMIZER
# ----------------------------------------------------------------------
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
)