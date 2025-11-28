"""
Baseline_00: PointPillars (Reference Method)
Purpose: Compare our adaptive method against PointPillars baseline
Expected: ~72-74% 3D AP@0.7 (standard KITTI Car performance)
Note: Simplified to work with 5-epoch comparison
"""

_base_ = ['../pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py']

# Override dataset path to match your setup
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Fix db_sampler path
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=['Car'],
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    backend_args=None)

# Update pipeline with correct db_sampler
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    dict(type='ObjectRangeFilter', point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

# Use same training config for fair comparison
train_cfg = dict(
    max_epochs=80,
    val_interval=5
)

# Same batch size and workers as baselines
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline))
)

# Update evaluation
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox'
)

