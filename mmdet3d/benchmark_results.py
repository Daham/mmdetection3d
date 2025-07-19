# benchmark_results.py
# Script to benchmark inference speed (FPS) and detection accuracy (mAP)

import os
import time
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.apis import init_model, inference_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmengine.evaluator import wrap_evaluator

# --------- User settings ---------
# Path to your config and checkpoint
CONFIG_PATH = 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car-SECOND.py'
CHECKPOINT_PATH = 'work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-car/latest.pth'
# Number of samples to use for speed test
WARMUP_ITERS = 20
TEST_ITERS = 200
# Batch size for speed test (should be 1)
BATCH_SIZE = 1
# GPUs
DEVICE = 'cuda:0'
# ----------------------------------

def benchmark_speed(model, dataloader):
    """Measure average FPS over TEST_ITERS samples"""
    # Warm up
    print(f"Warming up for {WARMUP_ITERS} iterations...")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= WARMUP_ITERS:
                break
            _ = model(return_loss=False, **data)
    torch.cuda.synchronize()

    # Timed run
    print(f"Running timed inference for {TEST_ITERS} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= TEST_ITERS:
                break
            _ = model(return_loss=False, **data)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    fps = TEST_ITERS / elapsed
    print(f"Average FPS over {TEST_ITERS} samples: {fps:.2f}")
    return fps


def benchmark_accuracy(model, dataset):
    """Run evaluation on KITTI validation split and print mAP results."""
    # Wrap evaluator for KITTI metrics
    evaluator = wrap_evaluator(
        dict(type='KittiMetric', ann_file=dataset.metainfo['ann_file'], backend_args=None)
    )
    # Collect predictions
    results = []
    model.eval()
    with torch.no_grad():
        for data in build_dataloader(
            dataset, samples_per_gpu=BATCH_SIZE, workers_per_gpu=2,
            dist=False, shuffle=False):
            result = model(return_loss=False, **data)
            results.extend(result)
    # Evaluate
    metrics = evaluator(dataset, results)
    print("Detection Accuracy (KITTI 3D mAP):")
    for cls, vals in metrics.items():
        # metrics dict: {'Car_easy': .,.}
        print(f"  {cls}: {vals:.2f}")
    return metrics


def main():
    # initialize 3D detection scope
    init_default_scope('mmdet3d')

    # Load config and model
    cfg = Config.fromfile(CONFIG_PATH)
    model = init_model(cfg, CHECKPOINT_PATH, device=DEVICE)

    # Build dataset and dataloader for validation
    dataset = build_dataset(cfg.test_dataloader.dataset)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=BATCH_SIZE, workers_per_gpu=2,
        dist=False, shuffle=False)

    # Benchmark speed
    fps = benchmark_speed(model, dataloader)

    # Benchmark accuracy
    metrics = benchmark_accuracy(model, dataset)

    # Summary
    print("\n=== Benchmark Summary ===")
    print(f"FPS: {fps:.2f}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

if __name__ == '__main__':
    main()