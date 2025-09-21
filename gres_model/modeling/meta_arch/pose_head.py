"""
Advanced Pose Estimation Head for MultitaskGRES.
Uses sophisticated transformer-based architecture for human pose estimation.
"""

# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Dict, List, Tuple
import math

from detectron2.layers import Linear
from .transformer_decoder.referring_transformer_decoder import (
    SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
)


class KeypointTransformerDecoder(nn.Module):
    """
    Sophisticated keypoint decoder with transformer layers and pose-specific attention.
    Uses anatomical structure awareness and hierarchical keypoint prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_keypoints: int = 17,
        num_pose_layers: int = 4,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
        pose_embed_dim: int = 256,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_keypoints = num_keypoints
        self.num_pose_layers = num_pose_layers
        self.pose_embed_dim = pose_embed_dim
        
        # Keypoint-specific embeddings (one for each COCO keypoint)
        self.keypoint_queries = nn.Embedding(num_keypoints, hidden_dim)
        self.keypoint_pos_embed = nn.Embedding(num_keypoints, hidden_dim)
        
        # Anatomical structure embeddings
        self.anatomical_groups = self._create_anatomical_groups()
        self.group_embeddings = nn.ModuleDict()
        for group_name, keypoint_ids in self.anatomical_groups.items():
            self.group_embeddings[group_name] = nn.Embedding(len(keypoint_ids), hidden_dim)
        
        # Multi-layer transformer for pose refinement
        self.pose_self_attn_layers = nn.ModuleList()
        self.pose_cross_attn_layers = nn.ModuleList()
        self.pose_anatomical_attn_layers = nn.ModuleList()
        self.pose_ffn_layers = nn.ModuleList()
        
        for _ in range(num_pose_layers):
            # Standard self and cross attention
            self.pose_self_attn_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            
            self.pose_cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            
            # Anatomical structure attention
            self.pose_anatomical_attn_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads // 2,  # Fewer heads for anatomical attention
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            
            self.pose_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
        
        # Language attention for pose (pose-specific RLA)
        self.pose_lang_attn = CrossAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        
        # Language projection
        self.lang_proj_pose = Linear(768, hidden_dim, False)
        self.lang_weight_pose = nn.Parameter(torch.tensor(0.0))
        
        # Hierarchical keypoint prediction layers
        self.coordinate_prediction_layers = nn.ModuleList()
        self.visibility_prediction_layers = nn.ModuleList()
        self.confidence_prediction_layers = nn.ModuleList()
        
        for i in range(num_pose_layers):
            # Coordinate regression (x, y)
            self.coordinate_prediction_layers.append(
                MLP(hidden_dim, pose_embed_dim, 2, 3)
            )
            
            # Visibility classification
            self.visibility_prediction_layers.append(
                MLP(hidden_dim, hidden_dim, 1, 2)
            )
            
            # Keypoint confidence
            self.confidence_prediction_layers.append(
                MLP(hidden_dim, hidden_dim, 1, 2)
            )
        
        # Pose quality assessment (similar to nt_embed in segmentation)
        self.pose_quality = MLP(hidden_dim, hidden_dim, 1, 2)
        
        # Anatomical consistency checker
        self.anatomical_consistency = nn.ModuleDict()
        for group_name in self.anatomical_groups.keys():
            self.anatomical_consistency[group_name] = MLP(hidden_dim, hidden_dim, 1, 2)
        
        # Spatial coordinate normalization
        self.coord_normalizer = nn.LayerNorm(2)
        
        # Keypoint relationship modeling (skeleton connections)
        self.skeleton_connections = self._create_skeleton_connections()
        self.connection_weights = nn.Parameter(torch.ones(len(self.skeleton_connections)))
        
        # Initialize weights
        self._reset_parameters()
    
    def _create_anatomical_groups(self) -> Dict[str, List[int]]:
        """Define anatomical groups for structured attention."""
        return {
            "head": [0, 1, 2, 3, 4],  # nose, eyes, ears
            "torso": [5, 6, 11, 12],  # shoulders, hips
            "left_arm": [5, 7, 9],    # left shoulder, elbow, wrist
            "right_arm": [6, 8, 10],  # right shoulder, elbow, wrist
            "left_leg": [11, 13, 15], # left hip, knee, ankle
            "right_leg": [12, 14, 16] # right hip, knee, ankle
        }
    
    def _create_skeleton_connections(self) -> List[Tuple[int, int]]:
        """Define skeleton connections between keypoints."""
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso connections
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10), # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)   # Right leg
        ]
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for language projection
        nn.init.zeros_(self.lang_proj_pose.weight)
        
        # Initialize connection weights
        nn.init.constant_(self.connection_weights, 1.0)
    
    def forward(
        self,
        shared_features: torch.Tensor,  # From segmentation transformer
        lang_feat: torch.Tensor,        # Language features
        mask_features: torch.Tensor,    # Mask features for spatial grounding
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            shared_features: (seq_len, batch, hidden_dim) - shared query features
            lang_feat: (batch, 768, seq_len) - language features from BERT
            mask_features: (batch, mask_dim, H, W) - mask features
        
        Returns:
            Dictionary with pose predictions at multiple layers
        """
        batch_size = shared_features.shape[1]
        
        # Initialize keypoint queries
        keypoint_queries = self.keypoint_queries.weight.unsqueeze(1).repeat(1, batch_size, 1)
        keypoint_pos = self.keypoint_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Store predictions from each layer
        all_coordinate_predictions = []
        all_visibility_predictions = []
        all_confidence_predictions = []
        
        current_features = keypoint_queries
        
        # Multi-layer pose transformer
        for layer_idx in range(self.num_pose_layers):
            # Self attention among keypoints (anatomical structure awareness)
            current_features = self.pose_self_attn_layers[layer_idx](
                current_features,
                query_pos=keypoint_pos
            )
            
            # Cross attention with shared visual features
            current_features = self.pose_cross_attn_layers[layer_idx](
                current_features,
                shared_features,
                query_pos=keypoint_pos,
                pos=None
            )
            
            # Anatomical group attention
            anatomical_enhanced = self._apply_anatomical_attention(
                current_features, layer_idx
            )
            current_features = current_features + anatomical_enhanced * 0.1
            
            # Language attention (Pose-specific RLA)
            if layer_idx == 0:  # Apply language attention in first layer
                lang_feat_proj = self.lang_proj_pose(lang_feat.permute(0, 2, 1))
                lang_attention = self.pose_lang_attn(
                    current_features,
                    lang_feat_proj.permute(1, 0, 2)
                )
                lang_attention = lang_attention * F.sigmoid(self.lang_weight_pose)
                current_features = current_features + lang_attention * 0.1
            
            # FFN layer
            current_features = self.pose_ffn_layers[layer_idx](current_features)
            
            # Generate predictions for this layer
            layer_features = current_features.transpose(0, 1)  # (batch, num_keypoints, hidden_dim)
            
            # Coordinate prediction
            coord_pred = self.coordinate_prediction_layers[layer_idx](layer_features)
            coord_pred = self.coord_normalizer(coord_pred)  # Normalize coordinates
            all_coordinate_predictions.append(coord_pred)
            
            # Visibility prediction
            vis_pred = self.visibility_prediction_layers[layer_idx](layer_features)
            all_visibility_predictions.append(vis_pred)
            
            # Confidence prediction
            conf_pred = self.confidence_prediction_layers[layer_idx](layer_features)
            all_confidence_predictions.append(conf_pred)
        
        # Combine coordinates and visibility into final format
        final_coordinates = all_coordinate_predictions[-1]  # (batch, num_keypoints, 2)
        final_visibility = all_visibility_predictions[-1]   # (batch, num_keypoints, 1)
        final_confidence = all_confidence_predictions[-1]   # (batch, num_keypoints, 1)
        
        # Stack into (batch, num_keypoints, 3) format
        final_keypoints = torch.cat([final_coordinates, final_visibility], dim=-1)
        
        # Pose quality assessment
        final_features = current_features.transpose(0, 1)  # (batch, num_keypoints, hidden_dim)
        pose_quality = self.pose_quality(final_features.mean(dim=1))  # Global average
        
        # Anatomical consistency scores
        anatomical_scores = {}
        for group_name, keypoint_ids in self.anatomical_groups.items():
            group_features = final_features[:, keypoint_ids, :].mean(dim=1)
            anatomical_scores[group_name] = self.anatomical_consistency[group_name](group_features)
        
        # Skeleton consistency (enforce anatomical constraints)
        skeleton_consistency = self._compute_skeleton_consistency(final_coordinates)
        
        return {
            "pred_keypoints": final_keypoints,                      # Final keypoint predictions
            "pred_coordinates": final_coordinates,                  # Just coordinates
            "pred_visibility": final_visibility,                    # Just visibility
            "pred_confidence": final_confidence,                    # Keypoint confidence
            "all_coordinate_predictions": all_coordinate_predictions, # All layers
            "all_visibility_predictions": all_visibility_predictions, # All layers
            "pose_quality": pose_quality,                           # Overall pose quality
            "anatomical_scores": anatomical_scores,                 # Per-group scores
            "skeleton_consistency": skeleton_consistency,           # Skeleton constraint score
            "pose_features": final_features,                        # Final pose features
        }
    
    def _apply_anatomical_attention(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply anatomical group attention."""
        batch_size = features.shape[1]
        enhanced_features = torch.zeros_like(features)
        
        for group_name, keypoint_ids in self.anatomical_groups.items():
            if len(keypoint_ids) > 1:
                group_features = features[keypoint_ids, :, :]  # (group_size, batch, hidden_dim)
                
                # Self-attention within anatomical group
                group_enhanced, _ = self.pose_anatomical_attn_layers[layer_idx].multihead_attn(
                    group_features, group_features, group_features
                )
                
                enhanced_features[keypoint_ids, :, :] = group_enhanced
        
        return enhanced_features
    
    def _compute_skeleton_consistency(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute skeleton consistency score based on anatomical constraints."""
        batch_size = coordinates.shape[0]
        consistency_scores = []
        
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx < coordinates.shape[1] and end_idx < coordinates.shape[1]:
                start_coords = coordinates[:, start_idx, :]  # (batch, 2)
                end_coords = coordinates[:, end_idx, :]      # (batch, 2)
                
                # Compute distance
                distance = torch.norm(end_coords - start_coords, dim=1)  # (batch,)
                consistency_scores.append(distance)
        
        if consistency_scores:
            consistency_tensor = torch.stack(consistency_scores, dim=1)  # (batch, num_connections)
            # Weight by learned connection importance
            weighted_consistency = consistency_tensor * self.connection_weights.unsqueeze(0)
            return weighted_consistency.mean(dim=1, keepdim=True)  # (batch, 1)
        else:
            return torch.zeros(batch_size, 1, device=coordinates.device)


class AdvancedPoseHead(nn.Module):
    """
    Advanced Pose Head with transformer decoder and sophisticated prediction mechanisms.
    Matches the complexity of the segmentation head with pose-specific enhancements.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_keypoints: int = 17,
        hidden_dim: int = 256,
        num_pose_layers: int = 4,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pose_loss_type: str = "mse",  # "mse", "l1", "smooth_l1"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.pose_loss_type = pose_loss_type
        
        # Input projection if needed
        self.input_proj = None
        if input_dim != hidden_dim:
            self.input_proj = Linear(input_dim, hidden_dim)
        
        # Core pose transformer decoder
        self.pose_decoder = KeypointTransformerDecoder(
            hidden_dim=hidden_dim,
            num_keypoints=num_keypoints,
            num_pose_layers=num_pose_layers,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Multi-scale pose estimation
        self.multi_scale_pose = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, num_keypoints * 2, 3) for _ in range(3)
        ])
        
        # Hierarchical pose classification (coarse pose categories)
        self.pose_classifier = MLP(hidden_dim, hidden_dim, 8, 3)  # 8 basic pose categories
        
        # Pose uncertainty estimation
        self.pose_uncertainty = MLP(hidden_dim, hidden_dim, num_keypoints, 2)
        
        # Feature enhancement with pose-specific augmentation
        self.pose_feature_enhancer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Temporal consistency module (for video sequences)
        self.temporal_consistency = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout
        )
    
    def forward(
        self,
        query_features: torch.Tensor,
        lang_feat: Optional[torch.Tensor] = None,
        mask_features: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: (batch, num_queries, input_dim) - Query features from shared transformer
            lang_feat: (batch, 768, seq_len) - Language features
            mask_features: (batch, mask_dim, H, W) - Mask features
            temporal_features: (batch, seq_len, hidden_dim) - Temporal features for video
        
        Returns:
            Dictionary with comprehensive pose outputs
        """
        batch_size, num_queries, _ = query_features.shape
        
        # Project input features if needed
        if self.input_proj is not None:
            query_features = self.input_proj(query_features)
        
        # Enhance features with pose-specific processing
        enhanced_features = self.pose_feature_enhancer(query_features)
        
        # Apply temporal consistency if available
        if temporal_features is not None:
            temporal_output, _ = self.temporal_consistency(temporal_features)
            enhanced_features = enhanced_features + temporal_output.mean(dim=1, keepdim=True)
        
        # Prepare for transformer decoder
        transformer_input = enhanced_features.transpose(0, 1)
        
        # Apply pose transformer decoder
        if lang_feat is not None and mask_features is not None:
            pose_outputs = self.pose_decoder(
                transformer_input, lang_feat, mask_features
            )
        else:
            # Fallback for testing without language/mask features
            pose_outputs = self._simple_forward(enhanced_features)
        
        # Multi-scale pose estimation
        multi_scale_poses = []
        for pose_head in self.multi_scale_pose:
            pose_pred = pose_head(enhanced_features)  # (batch, num_queries, num_keypoints*2)
            pose_pred = pose_pred.view(batch_size, num_queries, self.num_keypoints, 2)
            multi_scale_poses.append(pose_pred)
        
        # Hierarchical pose classification
        pose_categories = self.pose_classifier(enhanced_features.mean(dim=1))  # (batch, 8)
        
        # Pose uncertainty estimation
        pose_uncertainty = self.pose_uncertainty(enhanced_features.mean(dim=1))  # (batch, num_keypoints)
        
        # Combine all outputs
        final_outputs = {
            **pose_outputs,
            "multi_scale_poses": multi_scale_poses,
            "pose_categories": pose_categories,
            "pose_uncertainty": pose_uncertainty,
            "enhanced_features": enhanced_features,
        }
        
        return final_outputs
    
    def _simple_forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified forward for testing without language features."""
        batch_size, num_queries, hidden_dim = features.shape
        
        # Basic keypoint prediction
        keypoint_pred = self.multi_scale_pose[0](features)
        keypoint_pred = keypoint_pred.view(batch_size, num_queries, self.num_keypoints, 2)
        
        # Add dummy visibility
        visibility = torch.ones(batch_size, num_queries, self.num_keypoints, 1, device=features.device)
        keypoints = torch.cat([keypoint_pred, visibility], dim=-1)
        
        return {
            "pred_keypoints": keypoints,
            "pred_coordinates": keypoint_pred,
            "pred_visibility": visibility,
            "pose_quality": torch.zeros(batch_size, 1, device=features.device),
            "pose_features": features,
        }


def build_advanced_pose_head(cfg, input_dim: int):
    """Build advanced pose head from configuration."""
    return AdvancedPoseHead(
        input_dim=input_dim,
        num_keypoints=cfg.get("NUM_KEYPOINTS", 17),
        hidden_dim=cfg.get("POSE_HIDDEN_DIM", 256),
        num_pose_layers=cfg.get("NUM_POSE_LAYERS", 4),
        nheads=cfg.get("POSE_NHEADS", 8),
        dim_feedforward=cfg.get("POSE_DIM_FEEDFORWARD", 2048),
        dropout=cfg.get("POSE_DROPOUT", 0.1),
        pose_loss_type=cfg.get("POSE_LOSS_TYPE", "mse"),
    )