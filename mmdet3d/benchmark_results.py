# benchmark_results.py
import os
import time
import argparse
import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.datasets import build_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark inference speed & KITTI mAP')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--max-iter',
        type=int,
        default=None,
        help='Only benchmark this many samples (for quick test)')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load model
    model = init_model(
        args.config, args.checkpoint, device=args.device)

    # Grab its config to build val dataloader
    cfg = model.cfg
    val_dataset = build_dataset(cfg.test_dataloader.dataset)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 2) Inference + timing
    latencies = []
    results = []
    model.eval()
    torch.cuda.synchronize()
    for i, data in enumerate(val_loader):
        if args.max_iter and i >= args.max_iter:
            break
        # extract raw points (list of point arrays)
        pts = data['points'][0].numpy()
        torch.cuda.synchronize()
        t0 = time.time()
        res = inference_detector(model, pts)
        torch.cuda.synchronize()
        t1 = time.time()
        latencies.append(t1 - t0)
        results.append(res)

    # 3) Speed metrics
    lat_ms = float(np.mean(latencies) * 1000)
    fps = len(latencies) / sum(latencies)
    print(f'Inference: {lat_ms:.1f} ms avg / {fps:.1f} FPS '
          f'(over {len(latencies)} samples)')

    # 4) KITTI eval
    eval_res = val_dataset.evaluate(
        results,
        metric=['bbox', 'bev', '3d'],
        logger=None  # print to stdout
    )
    print('\n=== KITTI mAP ===')
    for k, v in eval_res.items():
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()
