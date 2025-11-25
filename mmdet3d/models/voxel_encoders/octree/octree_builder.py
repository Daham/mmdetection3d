"""
Adaptive octree builder with learned splitting criteria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .octree_node import OctreeNode


class AdaptiveOctreeBuilder(nn.Module):
    """
    Build adaptive octree with learned splitting criteria
    """
    
    def __init__(
        self,
        feat_channels: int = 128,
        max_depth: int = 6,
        min_points_per_voxel: int = 5,
        max_points_per_voxel: int = 100,
        split_temperature: float = 1.0,
        learnable_split: bool = True
    ):
        super().__init__()
        
        self.feat_channels = feat_channels
        self.max_depth = max_depth
        self.min_points_per_voxel = min_points_per_voxel
        self.max_points_per_voxel = max_points_per_voxel
        self.split_temperature = split_temperature
        self.learnable_split = learnable_split
        
        if learnable_split:
            # Neural network to predict split decision
            self.split_predictor = nn.Sequential(
                nn.Linear(feat_channels + 4, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)  # Binary decision
            )
    
    def compute_node_statistics(
        self,
        points: torch.Tensor,
        node: OctreeNode
    ) -> torch.Tensor:
        """
        Compute statistical features for split decision
        
        Returns:
            stats: (4,) [normalized_num_points, density, variance, depth_ratio]
        """
        num_points = len(points)
        volume = node.get_volume()
        density = num_points / volume if volume > 0 else 0.0
        
        if num_points > 1:
            spatial_var = torch.var(points[:, :3]).item()
        else:
            spatial_var = 0.0
        
        depth_ratio = node.depth / self.max_depth
        
        stats = torch.tensor([
            num_points / self.max_points_per_voxel,  # Normalized count
            min(density, 1.0),  # Normalized density
            spatial_var,
            depth_ratio
        ], device=points.device)
        
        return stats
    
    def should_split(
        self,
        node: OctreeNode,
        points: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Decide whether to split node
        
        Returns:
            should_split: Boolean decision
            split_score: Confidence score
        """
        # Hard constraints
        if node.depth >= self.max_depth:
            return False, 0.0
        
        if len(points) < self.min_points_per_voxel:
            return False, 0.0
        
        if not self.learnable_split:
            # Heuristic: split if too many points
            split = len(points) > self.max_points_per_voxel
            return split, 1.0 if split else 0.0
        
        # Learned splitting criterion
        stats = self.compute_node_statistics(points, node)
        decision_input = torch.cat([features, stats])
        
        split_logit = self.split_predictor(decision_input.unsqueeze(0)).squeeze(0)
        split_prob = torch.sigmoid(split_logit).item()
        
        if self.training:
            # Gumbel-Softmax for differentiable sampling
            gumbel_probs = F.gumbel_softmax(
                torch.stack([1 - split_logit, split_logit]),
                tau=self.split_temperature,
                hard=True
            )
            should_split = bool(gumbel_probs[1].item() > 0.5)
        else:
            # Deterministic at inference
            should_split = split_prob > 0.5
        
        return should_split, split_prob
    
    def build(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        root_bounds: torch.Tensor
    ) -> Tuple[OctreeNode, Dict]:
        """
        Build adaptive octree
        
        Args:
            points: (N, 4) xyz + intensity
            point_features: (N, C) encoded features
            root_bounds: (6,) scene bounds
        
        Returns:
            root: Root octree node
            stats: Build statistics
        """
        # Initialize root
        device = points.device
        root = OctreeNode(
            bounds=root_bounds,
            depth=0,
            max_depth=self.max_depth
        )
        root.point_indices = torch.arange(len(points), device=device)
        
        # BFS traversal for octree construction
        queue = [root]
        total_splits = 0
        total_evaluated = 0
        depth_distribution = torch.zeros(self.max_depth + 1, device=device)
        
        while queue:
            node = queue.pop(0)
            total_evaluated += 1
            
            # Get points in this node
            if node.point_indices is None or len(node.point_indices) == 0:
                continue
            
            node_points = points[node.point_indices]
            node_features = point_features[node.point_indices]
            
            # Aggregate features
            aggregated_feat = torch.mean(node_features, dim=0)
            node.features = aggregated_feat
            
            # Decide split
            should_split, split_score = self.should_split(
                node, node_points, aggregated_feat
            )
            
            if should_split:
                total_splits += 1
                children = node.subdivide()
                
                # Distribute points to children
                for child in children:
                    child_mask = self._points_in_bounds(
                        node_points, child.bounds
                    )
                    child.point_indices = node.point_indices[child_mask]
                    
                    if len(child.point_indices) > 0:
                        queue.append(child)
            else:
                # Leaf node
                depth_distribution[node.depth] += 1
        
        stats = {
            'total_nodes_evaluated': total_evaluated,
            'total_splits': total_splits,
            'depth_distribution': depth_distribution.cpu().numpy().tolist(),
            'num_leaves': int(depth_distribution.sum().item())
        }
        
        return root, stats
    
    def _points_in_bounds(
        self,
        points: torch.Tensor,
        bounds: torch.Tensor
    ) -> torch.Tensor:
        """Check which points are within bounds"""
        xyz = points[:, :3]
        min_bounds = bounds[:3]
        max_bounds = bounds[3:]
        
        mask = (
            (xyz[:, 0] >= min_bounds[0]) & (xyz[:, 0] < max_bounds[0]) &
            (xyz[:, 1] >= min_bounds[1]) & (xyz[:, 1] < max_bounds[1]) &
            (xyz[:, 2] >= min_bounds[2]) & (xyz[:, 2] < max_bounds[2])
        )
        
        return mask
