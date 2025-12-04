#!/usr/bin/env python3
"""Debug script to check inference results"""

import sys
sys.path.insert(0, '/home/daham/mmdetection_project/mmdetection3d')

from mmdet3d.apis import LidarDet3DInferencer
import numpy as np

print("Loading model...")
inferencer = LidarDet3DInferencer(
    model='configs/second/baseline_03_adaptive_multiscale_learnable.py',
    weights='work_dirs/method3_with_fix_5epochs/epoch_5.pth',
    device='cuda:0'
)

print("\nRunning inference on sample 000000...")
pcd_path = '/home/daham/mmdetection_project/dataset/KITTI/training/velodyne/000000.bin'

result = inferencer(
    inputs=dict(points=pcd_path),
    pred_score_thr=0.3,
    no_save_vis=True,
    no_save_pred=True,
    out_dir=''
)

print(f"\nResult type: {type(result)}")
print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

if isinstance(result, dict) and 'predictions' in result:
    pred = result['predictions'][0]
    print(f"\nPrediction type: {type(pred)}")
    
    if isinstance(pred, dict):
        print(f"Prediction keys: {pred.keys()}")
        
        # Check for different possible keys
        for key in ['pred_instances_3d', 'bboxes_3d', 'scores_3d', 'labels_3d']:
            if key in pred:
                print(f"\n{key}: {type(pred[key])}")
                if hasattr(pred[key], 'shape'):
                    print(f"  Shape: {pred[key].shape}")
                elif hasattr(pred[key], '__len__'):
                    print(f"  Length: {len(pred[key])}")
                
                # Print first few elements
                val = pred[key]
                if hasattr(val, 'tensor'):
                    print(f"  Tensor shape: {val.tensor.shape}")
                    print(f"  First element: {val.tensor[0] if len(val.tensor) > 0 else 'Empty'}")
                elif hasattr(val, '__getitem__'):
                    print(f"  First few: {val[:3] if len(val) > 0 else 'Empty'}")
    else:
        print(f"Prediction attributes: {dir(pred)}")
        
        if hasattr(pred, 'pred_instances_3d'):
            pred_inst = pred.pred_instances_3d
            print(f"\nPrediction instances type: {type(pred_inst)}")
            print(f"Attributes: {dir(pred_inst)}")
            
            if hasattr(pred_inst, 'bboxes_3d'):
                boxes = pred_inst.bboxes_3d
                print(f"\nBoxes type: {type(boxes)}")
                print(f"Number of boxes: {len(boxes)}")
                
                if hasattr(boxes, 'tensor'):
                    print(f"Boxes tensor shape: {boxes.tensor.shape}")
                    print(f"First box: {boxes.tensor[0] if len(boxes) > 0 else 'No boxes'}")
            
            if hasattr(pred_inst, 'scores_3d'):
                scores = pred_inst.scores_3d
                print(f"\nScores shape: {scores.shape if hasattr(scores, 'shape') else len(scores)}")
                print(f"Scores: {scores[:5] if len(scores) > 0 else 'No scores'}")
            
            if hasattr(pred_inst, 'labels_3d'):
                labels = pred_inst.labels_3d
                print(f"\nLabels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
                print(f"Labels: {labels[:5] if len(labels) > 0 else 'No labels'}")
else:
    print(f"\nFull result: {result}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
