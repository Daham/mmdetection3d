import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.models.layers.transformer import ConvBNPositionalEncoding
from .voxel_encoder import HardSimpleVFE

@MODELS.register_module()
class AdaptiveVFE(nn.Module):
    """Adaptive VFE with transformer-based attention for dynamic voxelization.

    Args:
        base_vfe_cfg (dict): Config for base VFE (e.g., HardSimpleVFE)
        embed_dims (int): Transformer embedding dimensions
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        pos_encoding_cfg (dict): Positional encoding config
        attention_threshold (float): Threshold for voxel fusion decisions
    """

    def __init__(self,
                 base_vfe_cfg=dict(type='HardSimpleVFE', num_features=4),
                 embed_dims=256,
                 num_heads=8,
                 num_layers=3,
                 pos_encoding_cfg=dict(input_channel=3, num_pos_feats=256),
                 attention_threshold=0.5,
                 voxel_size=[0.05, 0.05, 0.1],  # Add these parameters
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                 num_features=4
             ):
        super().__init__()

        print("----INSIDE INIT-----")

        # Store voxel parameters
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)

        # Create base VFE directly instead of using config
        self.base_vfe = HardSimpleVFE(num_features=num_features)

        # Base VFE for initial feature extraction
        self.base_vfe = MODELS.build(base_vfe_cfg)

        # Positional encoding for voxel coordinates
        self.pos_encoding = ConvBNPositionalEncoding(**pos_encoding_cfg)

        # Project voxel features to transformer dimensions
        self.feature_proj = nn.Linear(base_vfe_cfg.get('num_features', 4), embed_dims)

        # Transformer encoder for attention computation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Attention threshold for adaptive fusion
        self.attention_threshold = attention_threshold

        # Add output projection to match expected feature dimensions
        base_vfe_out_dim = 4  # HardSimpleVFE outputs 4 features
        self.output_proj = nn.Linear(embed_dims, base_vfe_out_dim)


    def forward(self, features, num_points, coors, *args, **kwargs):
        """
        Args:
            features: [N, M, C] voxel point features
            num_points: [N] number of points per voxel
            coors: [N, 4] voxel coordinates (batch_id, z, y, x)
        """
        batch_size = coors[:, 0].max().item() + 1

        # Step 1: Get base voxel features
        voxel_feats = self.base_vfe(features, num_points, coors)  # [N, C]

        # Step 2: Add positional encoding
        # Convert voxel coordinates to world coordinates for positional encoding
        voxel_centers = self.get_voxel_centers(coors)  # [N, 3]

        results = []
        attention_maps = []

        for b in range(batch_size):
            batch_mask = coors[:, 0] == b
            batch_feats = voxel_feats[batch_mask]  # [N_b, C]
            batch_centers = voxel_centers[batch_mask]  # [N_b, 3]

            if batch_feats.size(0) == 0:
                continue

            # Project features to transformer dimensions
            projected_feats = self.feature_proj(batch_feats)  # [N_b, embed_dims]

            # Add positional encoding
            pos_encoding = self.pos_encoding(batch_centers.unsqueeze(0))  # [1, embed_dims, N_b]
            pos_encoding = pos_encoding.transpose(1, 2)  # [1, N_b, embed_dims]

            # Combine features with positional encoding
            transformer_input = projected_feats.unsqueeze(0) + pos_encoding  # [1, N_b, embed_dims]

            # Step 3: Apply transformer to get attention patterns
            transformer_output = self.transformer(transformer_input)  # [1, N_b, embed_dims]

            # Extract attention map (you can modify this based on your fusion strategy)
            attention_weights = self.compute_attention_weights(transformer_output)
            attention_maps.append(attention_weights)

            # Step 4: Adaptive voxel fusion based on attention
            fused_feats, fused_coors = self.adaptive_fusion(
                batch_feats, coors[batch_mask], attention_weights
            )

            results.append((fused_feats, fused_coors))

        # Combine results from all batches
        final_features = torch.cat([r[0] for r in results], dim=0)
        final_coors = torch.cat([r[1] for r in results], dim=0)

        # Project back to original feature dimensions
        final_features = self.output_proj(final_features)

        return final_features

    def get_voxel_centers(self, coors):
        """Convert voxel indices to world coordinates."""
        centers = coors[:, 1:].float() * self.voxel_size.to(coors.device)
        centers += self.point_cloud_range[:3].to(coors.device)
        return centers

    def compute_attention_weights(self, transformer_output):
        """Extract attention weights for adaptive fusion decisions."""
        # You can implement different strategies here:
        # 1. Use attention weights from transformer layers
        # 2. Learn attention from transformer output
        # 3. Use similarity between neighboring voxels

        output = transformer_output.squeeze(0)  # [N_b, embed_dims]

        # Simple approach: compute pairwise similarities
        similarities = torch.mm(output, output.t())  # [N_b, N_b]
        attention_weights = torch.softmax(similarities, dim=-1)

        return attention_weights

    def adaptive_fusion(self, features, coors, attention_weights):
        """Fuse voxels based on attention weights."""
        # Implement your fusion strategy here
        # This is a simplified version - you'll need to implement proper clustering

        fusion_mask = attention_weights.max(dim=-1)[0] > self.attention_threshold

        # For now, return original features (implement your fusion logic)
        return features, coors
