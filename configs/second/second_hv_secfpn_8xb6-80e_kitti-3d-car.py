_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]


voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = '/home/daham/mmdetection_project/dataset/KITTI/' 

model = dict(
    voxel_encoder=dict(
        type='AdaptiveVFE',                                # use your custom encoder :contentReference[oaicite:0]{index=0}
        base_vfe_cfg=dict(
            type='HardSimpleVFE',                          # underlying VFE
            num_features=4                                 # matches your point features per voxel
        ),
        embed_dims=256,                                    # transformer embedding dim
        num_heads=8,                                       # multi‐head attention
        num_layers=3,                                      # transformer depth
        pos_encoding_cfg=dict(
            type='ConvBNPositionalEncoding',
            input_channel=3,                               # x,y,z center coords
            num_pos_feats=256                             # must match embed_dims :contentReference[oaicite:1]{index=1}
        ),
        attention_threshold=0.3,                           # similarity threshold for merging
        significance_percentile=0.5,                       # keep top‑50% “info‑heavy” voxels
        voxel_size=voxel_size,                            # reuse your config’s voxel_size
        point_cloud_range=point_cloud_range               # reuse your config’s point_cloud_range
    ),
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