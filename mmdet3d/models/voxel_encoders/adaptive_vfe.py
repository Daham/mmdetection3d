import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
# Assuming ConvBNPositionalEncoding is correctly imported and available
# If not, you might need to ensure its definition is accessible, e.g.:
# from mmdet3d.models.layers import ConvBNPositionalEncoding
# For this example, we'll assume it's found via mmdet3d.registry
# from .voxel_encoder import HardSimpleVFE # Not strictly needed if using MODELS.build

@MODELS.register_module()
class AdaptiveVFE(nn.Module):
    """Adaptive VFE with transformer-based attention for dynamic voxelization.

    Args:
        base_vfe_cfg (dict): Config for base VFE (e.g., HardSimpleVFE).
        embed_dims (int): Transformer embedding dimensions. Also the output
                          dimension of the positional encoding.
        num_heads (int): Number of attention heads in the transformer.
        num_layers (int): Number of transformer encoder layers.
        pos_encoding_cfg (dict): Positional encoding config.
                                 `num_pos_feats` should match `embed_dims`.
        attention_threshold (float): Similarity threshold for merging sparse voxels.
        voxel_size (list): Dimensions of a single voxel [vx, vy, vz] in meters.
        point_cloud_range (list): Min/max coordinates of the point cloud domain
                                  [x_min, y_min, z_min, x_max, y_max, z_max].
        significance_percentile (float): Percentile to determine information-heavy
                                         voxels (e.g., 0.7 means voxels with significance
                                         score above the 70th percentile are heavy).
    """

    def __init__(self,
                 base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
                 embed_dims=256,
                 num_heads=8,
                 num_layers=3,
                 pos_encoding_cfg=dict(type='ConvBNPositionalEncoding', input_channel=3, num_pos_feats=256),
                 attention_threshold=0.5,
                 voxel_size=[0.05, 0.05, 0.1],
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                 significance_percentile=0.7
             ):
        super().__init__()

        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)

        self.base_vfe = MODELS.build(base_vfe_cfg)

        # Ensure positional encoding output dimension matches embed_dims
        if 'num_pos_feats' in pos_encoding_cfg and pos_encoding_cfg['num_pos_feats'] != embed_dims:
            print(f"Warning: Overriding pos_encoding_cfg.num_pos_feats ({pos_encoding_cfg['num_pos_feats']}) "
                  f"to match embed_dims ({embed_dims}).")
            pos_encoding_cfg['num_pos_feats'] = embed_dims
        elif 'num_pos_feats' not in pos_encoding_cfg:
             pos_encoding_cfg['num_pos_feats'] = embed_dims


        self.pos_encoding = MODELS.build(pos_encoding_cfg)

        base_vfe_actual_out_features = base_vfe_cfg.get('num_features', 4) # Or determine dynamically if possible
        self.feature_proj = nn.Linear(base_vfe_actual_out_features, embed_dims)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 4, # Common practice
            batch_first=True # Our inputs will be (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.attention_threshold = attention_threshold
        self.significance_percentile = significance_percentile

        self.output_proj = nn.Linear(embed_dims, base_vfe_actual_out_features)

    def forward(self, features, num_points, coors, *args, **kwargs):
        """
        Args:
            features (Tensor): Raw point features within each voxel.
                               Shape [N_total_voxels, max_points_in_voxel, num_raw_point_features].
            num_points (Tensor): Number of actual points in each voxel.
                                 Shape [N_total_voxels].
            coors (Tensor): Coordinates of each voxel (batch_id, z_idx, y_idx, x_idx).
                            Shape [N_total_voxels, 4].
        Returns:
            Tensor: Fused voxel features. Shape [N_fused_voxels, base_vfe_out_features].
                    Returns empty tensor if no voxels are processed or all are fused away.
        """
        if coors.shape[0] == 0: # No voxels to process
            return torch.empty((0, self.output_proj.out_features), device=features.device, dtype=features.dtype)

        batch_size = coors[:, 0].max().item() + 1

        voxel_feats_base = self.base_vfe(features, num_points, coors)  # [N_total, C_base]
        voxel_centers_world = self.get_voxel_centers(coors)          # [N_total, 3] (world_xyz)

        results_fused_feats_embed = []
        results_fused_coors = []

        for b_idx in range(batch_size):
            batch_mask = coors[:, 0] == b_idx
            if not torch.any(batch_mask):
                continue

            current_batch_feats_base = voxel_feats_base[batch_mask]    # [N_b, C_base]
            current_batch_coors = coors[batch_mask]                  # [N_b, 4]
            current_batch_centers_world = voxel_centers_world[batch_mask] # [N_b, 3]

            N_b = current_batch_feats_base.size(0)
            if N_b == 0:
                continue

            projected_feats = self.feature_proj(current_batch_feats_base)  # [N_b, embed_dims]

            # current_batch_centers_world is [N_b, 3]
            # ConvBNPositionalEncoding expects (B, N, 3) and internally does permute(0, 2, 1) to get (B, 3, N)
            pos_encoding_input = current_batch_centers_world.unsqueeze(0)  # [1, N_b, 3]
            print(f"Shape of input to self.pos_encoding: {pos_encoding_input.shape}")

            pos_encoding = self.pos_encoding(pos_encoding_input)           # [1, embed_dims, N_b]

            pos_encoding = pos_encoding.transpose(1, 2)                    # [1, N_b, embed_dims]

            transformer_input = projected_feats.unsqueeze(0) + pos_encoding  # [1, N_b, embed_dims]
            print(f"Shape of transformer input: {transformer_input.shape}")


            transformer_output = self.transformer(transformer_input)           # [1, N_b, embed_dims]
            transformer_output_squeezed = transformer_output.squeeze(0)      # [N_b, embed_dims]

            # Compute similarity matrix (e.g., cosine similarity)
            similarity_matrix = self.compute_similarity_matrix(transformer_output_squeezed) # [N_b, N_b]

            fused_feats_embed_dim, fused_coors_batch = self.adaptive_fusion(
                transformer_output_squeezed,
                current_batch_coors,
                similarity_matrix
            )

            if fused_feats_embed_dim.size(0) > 0:
                results_fused_feats_embed.append(fused_feats_embed_dim)
                results_fused_coors.append(fused_coors_batch)

        if not results_fused_feats_embed:
             return torch.empty((0, self.output_proj.out_features), device=features.device, dtype=features.dtype)

        final_features_embed_dim = torch.cat(results_fused_feats_embed, dim=0)
        # final_coors = torch.cat(results_fused_coors, dim=0) # Coors are not returned by VFE

        final_features_base_dim = self.output_proj(final_features_embed_dim)

        return final_features_base_dim # Standard VFE returns only features

    def get_voxel_centers(self, coors):
        """Convert voxel indices (b,z,y,x) to world_xyz coordinates (voxel centers)."""
        voxel_indices_bzyx = coors.float()

        # self.voxel_size is [vx, vy, vz] e.g., [0.05, 0.05, 0.1]
        # self.point_cloud_range[:3] is [x_min, y_min, z_min]

        # X coordinate: (x_idx + 0.5) * vx + x_min
        x_centers = (voxel_indices_bzyx[:, 3] + 0.5) * self.voxel_size[0].to(coors.device) + \
                    self.point_cloud_range[0].to(coors.device)
        # Y coordinate: (y_idx + 0.5) * vy + y_min
        y_centers = (voxel_indices_bzyx[:, 2] + 0.5) * self.voxel_size[1].to(coors.device) + \
                    self.point_cloud_range[1].to(coors.device)
        # Z coordinate: (z_idx + 0.5) * vz + z_min
        z_centers = (voxel_indices_bzyx[:, 1] + 0.5) * self.voxel_size[2].to(coors.device) + \
                    self.point_cloud_range[2].to(coors.device)

        centers_world_xyz = torch.stack([x_centers, y_centers, z_centers], dim=1) # [N_total, 3]
        return centers_world_xyz

    def compute_similarity_matrix(self, features_embed_dim):
        """Compute pairwise similarity matrix from features.
        Args:
            features_embed_dim (Tensor): Voxel features [N_b, embed_dims].
        Returns:
            Tensor: Pairwise similarity matrix [N_b, N_b].
        """
        if features_embed_dim.shape[0] == 0:
            return torch.empty((0,0), device=features_embed_dim.device, dtype=features_embed_dim.dtype)
        # Cosine similarity for stability and bounded range [-1, 1]
        normed_features = features_embed_dim / (torch.norm(features_embed_dim, p=2, dim=1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(normed_features, normed_features.t())
        return similarity_matrix

    def adaptive_fusion(self, current_batch_features_embed, current_batch_coors, similarity_matrix):
        """Fuse voxels within a single batch based on similarity and significance.
        Args:
            current_batch_features_embed (Tensor): Features [N_b, embed_dims].
            current_batch_coors (Tensor): Voxel coordinates [N_b, 4] (b,z,y,x).
            similarity_matrix (Tensor): Pairwise similarity [N_b, N_b].
        Returns:
            Tuple[Tensor, Tensor]: fused_features_embed [N_fused, embed_dims],
                                   fused_coors [N_fused, 4].
        """
        N_b = current_batch_features_embed.shape[0]
        if N_b == 0:
            return torch.empty((0, current_batch_features_embed.shape[1]), device=current_batch_features_embed.device), \
                   torch.empty((0, current_batch_coors.shape[1]), dtype=current_batch_coors.dtype, device=current_batch_coors.device)

        # 1. Determine "Information-Heavy" Voxels
        # Significance score: sum of similarities (how much a voxel relates to others)
        # Higher score means more connections/relations.
        significance_score = similarity_matrix.sum(dim=1)

        if N_b > 1 :
            # Voxels with scores above this threshold are considered "information-heavy"
            significance_threshold_val = torch.quantile(significance_score.float(), self.significance_percentile)
        else: # Only one voxel, always keep it
            significance_threshold_val = significance_score.min().item() - 1e-6 # Ensure it's kept

        is_info_heavy = significance_score > significance_threshold_val

        final_features_list = []
        final_coors_list = []
        # Mask to track voxels already processed (either kept as info-heavy or merged)
        processed_mask = torch.zeros(N_b, dtype=torch.bool, device=current_batch_features_embed.device)

        # Add all information-heavy voxels first
        info_heavy_indices = torch.where(is_info_heavy)[0]
        if info_heavy_indices.numel() > 0:
            final_features_list.append(current_batch_features_embed[info_heavy_indices])
            final_coors_list.append(current_batch_coors[info_heavy_indices])
            processed_mask[info_heavy_indices] = True

        # For efficient neighbor lookup: map ZYX coordinates (in this batch) to their index
        coors_zyx_in_batch = current_batch_coors[:, 1:4] # Z, Y, X indices
        map_coor_tuple_to_idx_in_batch = {tuple(c.tolist()): i for i, c in enumerate(coors_zyx_in_batch)}

        # Iterate through voxels that are NOT info-heavy and NOT yet processed
        # These are candidates for merging.
        candidate_original_indices = torch.where(~processed_mask)[0]

        # Sort candidates to process them in a consistent order (e.g., by significance or index)
        # This can help make merging more deterministic, though not strictly necessary.
        # sorted_candidate_indices = candidate_original_indices[torch.argsort(significance_score[candidate_original_indices])]

        for i_idx_in_batch in candidate_original_indices:
            if processed_mask[i_idx_in_batch]: # Already processed (e.g. merged as a neighbor)
                continue

            current_voxel_coor_zyx = coors_zyx_in_batch[i_idx_in_batch]
            best_merge_neighbor_idx_in_batch = -1
            max_similarity_with_neighbor = -float('inf') # Using raw similarity

            # Define 3D neighbor offsets (6-connectivity)
            neighbor_offsets = torch.tensor([
                [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1] # dZ, dY, dX
            ], device=current_voxel_coor_zyx.device, dtype=current_voxel_coor_zyx.dtype)

            for offset in neighbor_offsets:
                prospective_neighbor_zyx_tuple = tuple((current_voxel_coor_zyx + offset).tolist())

                if prospective_neighbor_zyx_tuple in map_coor_tuple_to_idx_in_batch:
                    j_idx_in_batch = map_coor_tuple_to_idx_in_batch[prospective_neighbor_zyx_tuple]

                    # Check if this neighbor is a valid merge partner:
                    # - Not itself (implicitly handled by offset)
                    # - Not already processed
                    # - Not information-heavy (we want to merge sparse with sparse)
                    if not processed_mask[j_idx_in_batch] and \
                       not is_info_heavy[j_idx_in_batch]:

                        similarity = similarity_matrix[i_idx_in_batch, j_idx_in_batch]
                        if similarity > max_similarity_with_neighbor:
                            max_similarity_with_neighbor = similarity
                            best_merge_neighbor_idx_in_batch = j_idx_in_batch

            if best_merge_neighbor_idx_in_batch != -1 and max_similarity_with_neighbor > self.attention_threshold:
                # Found a suitable neighbor to merge with
                j_idx = best_merge_neighbor_idx_in_batch

                # Merge features (e.g., average)
                merged_feature = (current_batch_features_embed[i_idx_in_batch] + \
                                  current_batch_features_embed[j_idx]) / 2.0
                # Use coordinates of the current voxel (i_idx_in_batch) for the merged entity
                merged_coor = current_batch_coors[i_idx_in_batch]

                final_features_list.append(merged_feature.unsqueeze(0))
                final_coors_list.append(merged_coor.unsqueeze(0))
                processed_mask[i_idx_in_batch] = True
                processed_mask[j_idx] = True # Mark the neighbor as processed too
            else:
                # Cannot merge (no suitable neighbor or similarity too low), so keep this voxel as is
                final_features_list.append(current_batch_features_embed[i_idx_in_batch].unsqueeze(0))
                final_coors_list.append(current_batch_coors[i_idx_in_batch].unsqueeze(0))
                processed_mask[i_idx_in_batch] = True

        if not final_features_list: # Should only happen if N_b was 0 initially
            return torch.empty((0, current_batch_features_embed.shape[1]), device=current_batch_features_embed.device), \
                   torch.empty((0, current_batch_coors.shape[1]), dtype=current_batch_coors.dtype, device=current_batch_coors.device)

        final_fused_features_embed = torch.cat(final_features_list, dim=0)
        final_fused_coors = torch.cat(final_coors_list, dim=0)

        return final_fused_features_embed, final_fused_coors
