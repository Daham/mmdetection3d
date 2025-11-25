"""
Octree node structure for adaptive voxelization
"""

import torch
from typing import List, Optional


class OctreeNode:
    """
    Node in adaptive octree structure
    
    Attributes:
        bounds: (6,) tensor [x_min, y_min, z_min, x_max, y_max, z_max]
        depth: Current depth in tree
        max_depth: Maximum allowed depth
        center: (3,) voxel center
        size: (3,) voxel dimensions
        is_leaf: Whether this is a leaf node
        children: List of 8 child nodes (if subdivided)
        point_indices: Indices of points in this voxel
        features: Aggregated features for this voxel
    """
    
    def __init__(self, bounds: torch.Tensor, depth: int = 0, max_depth: int = 6):
        """
        Args:
            bounds: (6,) tensor [x_min, y_min, z_min, x_max, y_max, z_max]
            depth: Current depth in octree
            max_depth: Maximum allowed depth
        """
        self.bounds = bounds
        self.depth = depth
        self.max_depth = max_depth
        self.center = (bounds[:3] + bounds[3:]) / 2
        self.size = bounds[3:] - bounds[:3]
        self.is_leaf = True
        self.children: Optional[List['OctreeNode']] = None
        self.point_indices: Optional[torch.Tensor] = None
        self.features: Optional[torch.Tensor] = None
    
    def get_voxel_size(self) -> float:
        """Get voxel size (assuming cubic voxels)"""
        return self.size[0].item()
    
    def get_volume(self) -> float:
        """Get voxel volume"""
        return torch.prod(self.size).item()
    
    def subdivide(self) -> List['OctreeNode']:
        """
        Subdivide node into 8 children (octants)
        
        Returns:
            List of 8 child nodes
        """
        if self.depth >= self.max_depth:
            return []
        
        self.is_leaf = False
        children = []
        
        mid = self.center
        min_bounds = self.bounds[:3]
        max_bounds = self.bounds[3:]
        device = self.bounds.device
        
        # Create 8 octants (2x2x2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_min = torch.tensor([
                        mid[0] if i else min_bounds[0],
                        mid[1] if j else min_bounds[1],
                        mid[2] if k else min_bounds[2]
                    ], device=device)
                    
                    child_max = torch.tensor([
                        max_bounds[0] if i else mid[0],
                        max_bounds[1] if j else mid[1],
                        max_bounds[2] if k else mid[2]
                    ], device=device)
                    
                    child_bounds = torch.cat([child_min, child_max])
                    child = OctreeNode(
                        bounds=child_bounds,
                        depth=self.depth + 1,
                        max_depth=self.max_depth
                    )
                    children.append(child)
        
        self.children = children
        return children
    
    def contains_point(self, point: torch.Tensor) -> bool:
        """Check if point is within this node's bounds"""
        xyz = point[:3]
        return (
            (xyz >= self.bounds[:3]).all() and 
            (xyz < self.bounds[3:]).all()
        ).item()
    
    def get_all_leaves(self) -> List['OctreeNode']:
        """Recursively collect all leaf nodes"""
        if self.is_leaf:
            return [self]
        
        leaves = []
        if self.children:
            for child in self.children:
                leaves.extend(child.get_all_leaves())
        return leaves
    
    def count_nodes(self) -> int:
        """Count total nodes in subtree"""
        if self.is_leaf:
            return 1
        
        count = 1
        if self.children:
            for child in self.children:
                count += child.count_nodes()
        return count
