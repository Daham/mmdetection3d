# STANDARD SECOND CONFIG FOR BASELINE TEST
# Use this to verify that basic training works without our adaptive module

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Use standard HardSimpleVFE (no adaptive processing)
model = dict(
    voxel_encoder=dict(type='HardSimpleVFE'),  # Standard VFE
    bbox_head=dict(num_classes=1))

# Short test run with frequent logging
train_cfg = dict(max_epochs=2, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))

# Frequent logging to see progress
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1)
)
