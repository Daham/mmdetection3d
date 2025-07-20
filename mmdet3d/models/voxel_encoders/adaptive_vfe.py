# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet3d.registry import MODELS

# @MODELS.register_module()
# class AdaptiveVFE(nn.Module):
#     """
#     Adaptive Voxel Feature Encoder with:
#         1) Base VFE to extract initial voxel features
#         2) Early percentile-based pruning to select top-scoring voxels
#         3) Memory-efficient merging of dropped voxels via block-wise similarity
#         4) Linear projection to higher-dimensional embeddings
#         5) Positional encoding via MLP on voxel centers
#         6) Local self-attention using TransformerEncoder
#         7) Back-projection to original feature channels for compatibility
#     """
#     def __init__(
#         self,
#         base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
#         embed_dims=128,
#         num_heads=4,
#         num_layers=1,
#         keep_ratio=0.2,
#         merge_block_size=1024,
#         voxel_size=(0.05, 0.05, 0.1),
#         point_cloud_range=(0.0, -40.0, -3.0)
#     ):
#         super().__init__()
#         # 1) Base VFE
#         self.base_vfe = MODELS.build(base_vfe_cfg)
#         self.in_ch = getattr(self.base_vfe, 'out_channels', base_vfe_cfg.get('num_features'))

#         # 2) Project to embedding dimension
#         self.proj = nn.Linear(self.in_ch, embed_dims)
#         # 5) Positional encoding MLP
#         self.pos_mlp = nn.Sequential(
#             nn.Linear(3, embed_dims),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dims, embed_dims)
#         )
#         # 6) Transformer for self-attention
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dims,
#             nhead=num_heads,
#             dim_feedforward=embed_dims * 2,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # 7) Back-projection to original channels
#         self.back_proj = nn.Linear(embed_dims, self.in_ch)

#         # Pruning and merging parameters
#         self.keep_ratio = keep_ratio
#         self.merge_block_size = merge_block_size

#         # Buffers for coordinate conversion
#         self.register_buffer('voxel_size', torch.tensor(voxel_size).float())
#         self.register_buffer('pc_range', torch.tensor(point_cloud_range).float())

#     def forward(self, features, num_points, coors):
#         """
#         Args:
#             features (Tensor[N, M, C_in]): per-point inputs
#             num_points (Tensor[N]): valid points per voxel
#             coors (Tensor[N, 4]): voxel coords (batch, z, y, x)
#         Returns:
#             out_feats (Tensor[K, C_in]), pruned_coors (Tensor[K, 4])
#         """
#         # 1) Base VFE
#         base_feats = self.base_vfe(features, num_points, coors)  # [N, C_in]

#         # 2) Early pruning: keep only top-scoring voxels
#         scores = base_feats.norm(p=2, dim=1)  # [N]
#         N = scores.size(0)
#         K = max(int(N * self.keep_ratio), 1)
#         _, keep_idx = torch.topk(scores, K, sorted=False)
#         kept_feats = base_feats[keep_idx]      # [K, C_in]
#         kept_coors = coors[keep_idx]           # [K, 4]
#         # Determine dropped indices
#         all_idx = torch.arange(N, device=scores.device)
#         drop_idx = all_idx[~torch.isin(all_idx, keep_idx)]
#         dropped_feats = base_feats[drop_idx]   # [N-K, C_in]

#         print(f"[AdaptiveVFE] early prune: {N} -> {K} voxels, merging {dropped_feats.size(0)}")

#         # 3) Block-wise merging for memory efficiency
#         if dropped_feats.numel() > 0:
#             # normalized kept features for cosine similarity
#             kept_norm = F.normalize(kept_feats, dim=1)  # [K, C_in]
#             B = self.merge_block_size
#             for start in range(0, dropped_feats.size(0), B):
#                 end = min(start + B, dropped_feats.size(0))
#                 block = dropped_feats[start:end]           # [b, C_in]
#                 block_norm = F.normalize(block, dim=1)     # [b, C_in]
#                 # similarity [b, K]
#                 sims = torch.matmul(block_norm, kept_norm.t())
#                 best = sims.argmax(dim=1)                  # [b]
#                 # merge by averaging
#                 for i, ki in enumerate(best):
#                     kept_feats[ki] = (kept_feats[ki] + block[i]) * 0.5

#         # 4) Project to embedding dimension
#         proj_feats = self.proj(kept_feats)          # [K, D]

#         # 5) Positional encoding
#         centers = (
#             kept_coors[:, 1:].float() * self.voxel_size
#             + self.pc_range.unsqueeze(0)
#             + self.voxel_size.unsqueeze(0) * 0.5
#         )  # [K, 3]
#         pos_feats = self.pos_mlp(centers)           # [K, D]
#         fused = proj_feats + pos_feats              # [K, D]

#         # 6) Local self-attention
#         attn_out = self.transformer(fused.unsqueeze(0)).squeeze(0)  # [K, D]

#         # 7) Back-project to original channels
#         out_feats = self.back_proj(attn_out)        # [K, C_in]

#         return out_feats, kept_coors


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class AdaptiveVFE(nn.Module):
    def __init__(
        self,
        base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
        embed_dims=128,
        num_heads=4,
        num_layers=1,
        keep_ratio=0.2,
        merge_block_size=1024,
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0),
        split_point_thresh=60  # new
    ):
        super().__init__()
        self.base_vfe = MODELS.build(base_vfe_cfg)
        self.in_ch = getattr(self.base_vfe, 'out_channels', base_vfe_cfg.get('num_features'))
        self.proj = nn.Linear(self.in_ch, embed_dims)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.back_proj = nn.Linear(embed_dims, self.in_ch)

        self.keep_ratio = keep_ratio
        self.merge_block_size = merge_block_size
        self.split_point_thresh = split_point_thresh  # new

        self.register_buffer('voxel_size', torch.tensor(voxel_size).float())
        self.register_buffer('pc_range', torch.tensor(point_cloud_range).float())

    def forward(self, features, num_points, coors):
        base_feats = self.base_vfe(features, num_points, coors)  # [N, C_in]
        scores = base_feats.norm(p=2, dim=1)  # [N]
        N = scores.size(0)
        K = max(int(N * self.keep_ratio), 1)
        _, keep_idx = torch.topk(scores, K, sorted=False)
        kept_feats = base_feats[keep_idx]      # [K, C_in]
        kept_coors = coors[keep_idx]           # [K, 4]

        all_idx = torch.arange(N, device=scores.device)
        drop_idx = all_idx[~torch.isin(all_idx, keep_idx)]
        dropped_feats = base_feats[drop_idx]

        print(f"[AdaptiveVFE] early prune: {N} -> {K} voxels, merging {dropped_feats.size(0)}")

        # === Splitting step ===
        split_feats = []
        split_coors = []

        for i in range(kept_coors.shape[0]):
            if num_points[keep_idx[i]] > self.split_point_thresh:
                orig_coord = kept_coors[i]
                center = orig_coord[1:].float() * self.voxel_size + self.pc_range + self.voxel_size * 0.5
                offsets = torch.tensor([[dx, dy, dz] for dx in [0, 0.5] for dy in [0, 0.5] for dz in [0, 0.5]], device=features.device)
                new_coords = (orig_coord[None, :].repeat(8, 1))
                new_coords[:, 1:] += (offsets * 2).long()  # simulate octant offsets
                split_coors.append(new_coords)
                split_feats.append(kept_feats[i].unsqueeze(0).repeat(8, 1))  # use same feat initially

            else:
                split_feats.append(kept_feats[i].unsqueeze(0))
                split_coors.append(kept_coors[i].unsqueeze(0))

        kept_feats = torch.cat(split_feats, dim=0)
        kept_coors = torch.cat(split_coors, dim=0)

        # === Merging step ===
        if dropped_feats.numel() > 0:
            kept_norm = F.normalize(kept_feats, dim=1)  # [K, C_in]
            B = self.merge_block_size
            for start in range(0, dropped_feats.size(0), B):
                end = min(start + B, dropped_feats.size(0))
                block = dropped_feats[start:end]
                block_norm = F.normalize(block, dim=1)
                sims = torch.matmul(block_norm, kept_norm.t())
                best = sims.argmax(dim=1)
                for i, ki in enumerate(best):
                    kept_feats[ki] = (kept_feats[ki] + block[i]) * 0.5

        # === Projection & Positional Encoding ===
        proj_feats = self.proj(kept_feats)
        centers = kept_coors[:, 1:].float() * self.voxel_size + self.pc_range + self.voxel_size * 0.5
        pos_feats = self.pos_mlp(centers)
        fused = proj_feats + pos_feats

        # === Transformer Attention ===
        attn_out = self.transformer(fused.unsqueeze(0)).squeeze(0)

        # === Output ===
        out_feats = self.back_proj(attn_out)
        return out_feats, kept_coors
