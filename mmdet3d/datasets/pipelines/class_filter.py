from mmcv.transforms import BaseTransform

class ClassFilter(BaseTransform):
    def __init__(self, keep_classes):
        self.keep_classes = keep_classes

    def transform(self, results):
        if 'gt_names' in results:
            keep_mask = [name in self.keep_classes for name in results['gt_names']]
            results['gt_names'] = [n for n, k in zip(results['gt_names'], keep_mask) if k]
            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][keep_mask]
            if 'gt_labels_3d' in results:
                results['gt_labels_3d'] = results['gt_labels_3d'][keep_mask]
        return results
