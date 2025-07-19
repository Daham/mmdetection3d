import torch
import torch.nn as nn
import time
import logging
from mmdet3d.registry import MODELS

# Set up logger
# Use __name__ which is a standard Python practice
logger = logging.getLogger(__name__)

@MODELS.register_module()
class AdaptiveVFE(nn.Module):
    """Adaptive VFE with transformer-based attention for dynamic voxelization.
    ... (docstring remains the same) ...
    """

    # FIX 1: Corrected constructor name from _init_ to __init__
    def __init__(self,
                 base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
                 embed_dims=256,
                 num_heads=8,
                 num_layers=3,
                 pos_encoding_cfg=dict(type='ConvBNPositionalEncoding', input_channel=3, num_pos_feats=256),
                 attention_threshold=0.3,
                 significance_percentile=0.5,
                 voxel_size=[0.05, 0.05, 0.1],
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                ):
        super().__init__()

        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)

        self.base_vfe = MODELS.build(base_vfe_cfg)

        if 'num_pos_feats' in pos_encoding_cfg and pos_encoding_cfg['num_pos_feats'] != embed_dims:
            # FIX 2: Use logger instead of print for framework consistency
            logger.warning(f"Overriding pos_encoding_cfg.num_pos_feats ({pos_encoding_cfg['num_pos_feats']}) "
                           f"to match embed_dims ({embed_dims}).")
            pos_encoding_cfg['num_pos_feats'] = embed_dims
        elif 'num_pos_feats' not in pos_encoding_cfg:
            pos_encoding_cfg['num_pos_feats'] = embed_dims

        self.pos_encoding = MODELS.build(pos_encoding_cfg)

        # Assuming HardSimpleVFE output is num_features. A more robust way might be needed if base_vfe is complex.
        base_vfe_actual_out_features = base_vfe_cfg.get('num_features', 4)
        self.feature_proj = nn.Linear(base_vfe_actual_out_features, embed_dims)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 4,
            batch_first=True,
            dropout=0.1 # Common practice to add dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.attention_threshold = attention_threshold
        self.significance_percentile = significance_percentile

        self.output_proj = nn.Linear(embed_dims, base_vfe_actual_out_features)


    def forward(self, features, num_points, coors, *args, **kwargs):
        """
        Args:
            features (Tensor): Raw point features. [N_voxels, max_points, C_in]
            num_points (Tensor): Number of points in each voxel. [N_voxels]
            coors (Tensor): Voxel coordinates. [N_voxels, 4] (b, z, y, x)
        Returns:
            Tuple[Tensor, Tensor]:
                - Fused voxel features. [N_fused_voxels, C_out]
                - Fused voxel coordinates. [N_fused_voxels, 4]
        """
        if coors.shape[0] == 0:
            logger.debug("AdaptiveVFE: No voxels to process, returning empty tensors")
            out_features = self.output_proj.out_features
            return torch.empty((0, out_features), device=features.device, dtype=features.dtype), \
                   torch.empty((0, 4), device=coors.device, dtype=coors.dtype)

        batch_size = coors[:, 0].max().item() + 1
        
        # Base VFE processing
        voxel_feats_base = self.base_vfe(features, num_points, coors)
        voxel_centers_world = self.get_voxel_centers(coors)

        results_fused_feats_embed = []
        results_fused_coors = []
        
        total_original_voxels = coors.shape[0]

        for b_idx in range(batch_size):
            batch_mask = coors[:, 0] == b_idx
            if not torch.any(batch_mask):
                continue

            current_batch_feats_base = voxel_feats_base[batch_mask]
            current_batch_coors = coors[batch_mask]
            current_batch_centers_world = voxel_centers_world[batch_mask]

            N_b = current_batch_feats_base.size(0)
            if N_b == 0:
                continue

            projected_feats = self.feature_proj(current_batch_feats_base)
            
            # Positional encoding expects [B, N, C], here B=1, N=N_b, C=3
            pos_encoding_input = current_batch_centers_world.unsqueeze(0)
            # Output of ConvBNPositionalEncoding is [B, C_embed, N], need to transpose
            pos_encoding = self.pos_encoding(pos_encoding_input).transpose(1, 2)

            transformer_input = projected_feats.unsqueeze(0) + pos_encoding
            transformer_output = self.transformer(transformer_input).squeeze(0)

            similarity_matrix = self.compute_similarity_matrix(transformer_output)
            
            fused_feats_embed_dim, fused_coors_batch = self.adaptive_fusion(
                transformer_output,
                current_batch_coors,
                similarity_matrix
            )

            if fused_feats_embed_dim.size(0) > 0:
                results_fused_feats_embed.append(fused_feats_embed_dim)
                results_fused_coors.append(fused_coors_batch)

        if not results_fused_feats_embed:
            logger.warning("AdaptiveVFE: No voxels remained after fusion!")
            out_features = self.output_proj.out_features
            return torch.empty((0, out_features), device=features.device, dtype=features.dtype), \
                   torch.empty((0, 4), device=coors.device, dtype=coors.dtype)
        
        # Concatenate results from all batches
        final_features_embed_dim = torch.cat(results_fused_feats_embed, dim=0)
        final_fused_coors = torch.cat(results_fused_coors, dim=0)

        # Final projection to match expected output dimension
        final_features_base_dim = self.output_proj(final_features_embed_dim)
        
        total_fused_voxels = final_features_base_dim.shape[0]
        compression = total_fused_voxels / total_original_voxels if total_original_voxels > 0 else 0
        logger.debug(f"AdaptiveVFE: Voxel compression: {total_original_voxels} -> {total_fused_voxels} ({compression:.3f} ratio)")
        
        # FIX 3: Return both features and the corresponding new coordinates
        return final_features_base_dim, final_fused_coors


    # def get_voxel_centers(self, coors):
    #     # Move tensors to device once to avoid repeated calls
    #     device = coors.device
    #     voxel_size_dev = self.voxel_size.to(device)
    #     pc_range_min_dev = self.point_cloud_range[:3].to(device)

    #     voxel_indices_bzyx = coors.float()
        
    #     # Order is [x, y, z]
    #     offsets = torch.tensor([0.5], device=device)
        
    #     x_centers = (voxel_indices_bzyx[:, 3] + offsets) * voxel_size_dev[0] + pc_range_min_dev[0]
    #     y_centers = (voxel_indices_bzyx[:, 2] + offsets) * voxel_size_dev[1] + pc_range_min_dev[1]
    #     z_centers = (voxel_indices_bzyx[:, 1] + offsets) * voxel_size_dev[2] + pc_range_min_dev[2]

    #     return torch.stack([x_centers, y_centers, z_centers], dim=1)
    
    def get_voxel_centers(self, coors):
        """Calculates the center of voxels in real-world coordinates."""
        device = coors.device
        # Ensure voxel_size and pc_range are on the correct device
        voxel_size_dev = self.voxel_size.to(device)
        pc_range_min_dev = self.point_cloud_range[:3].to(device)

        # Get the z, y, x indices from coordinates
        voxel_indices_zyx = coors[:, 1:4].float()

        # Flip the order to x, y, z to match the convention of voxel_size and pc_range
        voxel_indices_xyz = torch.flip(voxel_indices_zyx, dims=[1])

        # Calculate all centers at once using broadcasting
        # Formula: (index + 0.5) * size + minimum_range
        centers = (voxel_indices_xyz + 0.5) * voxel_size_dev + pc_range_min_dev

        return centers

    def compute_similarity_matrix(self, features_embed_dim):
        if features_embed_dim.shape[0] == 0:
            return torch.empty((0,0), device=features_embed_dim.device, dtype=features_embed_dim.dtype)
        normed_features = torch.nn.functional.normalize(features_embed_dim, p=2, dim=1)
        return torch.mm(normed_features, normed_features.t())



    def adaptive_fusion(self, current_batch_features_embed, current_batch_coors, similarity_matrix):
        N_b = current_batch_features_embed.shape[0]
        if N_b == 0:
            return torch.empty((0, current_batch_features_embed.shape[1]), device=current_batch_features_embed.device), \
                   torch.empty((0, current_batch_coors.shape[1]), dtype=current_batch_coors.dtype, device=current_batch_coors.device)

        # Significance computation
        significance_score = similarity_matrix.sum(dim=1)
        if N_b > 1:
            significance_threshold_val = torch.quantile(significance_score.float(), self.significance_percentile)
        else:
            significance_threshold_val = significance_score.item() + 1.0

        is_info_heavy = significance_score > significance_threshold_val
        processed_mask = torch.zeros(N_b, dtype=torch.bool, device=current_batch_features_embed.device)

        # Directly keep all information-heavy voxels
        final_features_list = [current_batch_features_embed[is_info_heavy]]
        final_coors_list = [current_batch_coors[is_info_heavy]]
        processed_mask[is_info_heavy] = True

        # Process candidates for merging or keeping
        candidates_indices = torch.where(~processed_mask)[0]
        if candidates_indices.numel() > 0:
            coors_zyx_in_batch = current_batch_coors[:, 1:4]
            map_coor_tuple_to_idx_in_batch = {tuple(c.tolist()): i for i, c in enumerate(coors_zyx_in_batch)}

            for i_idx in candidates_indices:
                if processed_mask[i_idx]:
                    continue

                current_voxel_coor_zyx = coors_zyx_in_batch[i_idx]
                best_merge_neighbor_idx = -1
                max_similarity_with_neighbor = -float('inf')

                for offset in [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]:
                    neighbor_coor = tuple((current_voxel_coor_zyx + torch.tensor(offset, device=i_idx.device)).tolist())
                    if neighbor_coor in map_coor_tuple_to_idx_in_batch:
                        j_idx = map_coor_tuple_to_idx_in_batch[neighbor_coor]
                        if not processed_mask[j_idx]:
                            similarity = similarity_matrix[i_idx, j_idx]
                            if similarity > max_similarity_with_neighbor:
                                max_similarity_with_neighbor = similarity
                                best_merge_neighbor_idx = j_idx

                if best_merge_neighbor_idx != -1 and max_similarity_with_neighbor > self.attention_threshold:
                    j_idx = best_merge_neighbor_idx
                    merged_feature = (current_batch_features_embed[i_idx] + current_batch_features_embed[j_idx]) / 2.0
                    final_features_list.append(merged_feature.unsqueeze(0))
                    final_coors_list.append(current_batch_coors[i_idx].unsqueeze(0))
                    processed_mask[i_idx] = True
                    processed_mask[j_idx] = True
                else:
                    final_features_list.append(current_batch_features_embed[i_idx].unsqueeze(0))
                    final_coors_list.append(current_batch_coors[i_idx].unsqueeze(0))
                    processed_mask[i_idx] = True

        # Concatenate all gathered features and coordinates
        final_features = torch.cat(final_features_list, dim=0)
        final_coors = torch.cat(final_coors_list, dim=0)

        # FINAL SAFEGUARD: If the output is STILL empty, keep the single most significant voxel.
        if final_features.shape[0] == 0 and N_b > 0:
            logger.warning("AdaptiveVFE Safeguard (Post-Cat): All voxels were discarded, keeping the most significant one.")
            most_significant_idx = torch.argmax(significance_score)
            final_features = current_batch_features_embed[most_significant_idx].unsqueeze(0)
            final_coors = current_batch_coors[most_significant_idx].unsqueeze(0)

        return final_features, final_coors