_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# MINIMAL changes - only what's absolutely necessary
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Override training epochs
train_cfg = dict(max_epochs=5, val_interval=1)

# Override optimizer to match your research setup
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01)
)
