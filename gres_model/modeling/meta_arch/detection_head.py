"""
Advanced Detection Head for MultitaskGRES.
Uses sophisticated transformer-based architecture similar to the segmentation head.
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
from typing import Optional, Dict, List
import math

from detectron2.layers import Linear
from .transformer_decoder.referring_transformer_decoder import (
    SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
)


class DetectionTransformerDecoder(nn.Module):
    """
    Sophisticated detection decoder with transformer layers and attention mechanisms.
    Similar complexity to the segmentation head's MultiScaleMaskedReferringDecoder.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_detection_layers: int = 3,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
        num_detection_queries: int = 100,
        bbox_embed_dim: int = 256,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_detection_layers = num_detection_layers
        self.num_detection_queries = num_detection_queries
        
        # Detection-specific query embeddings
        self.detection_queries = nn.Embedding(num_detection_queries, hidden_dim)
        self.detection_pos_embed = nn.Embedding(num_detection_queries, hidden_dim)
        
        # Multi-layer transformer for detection refinement
        self.detection_self_attn_layers = nn.ModuleList()
        self.detection_cross_attn_layers = nn.ModuleList()
        self.detection_ffn_layers = nn.ModuleList()
        
        for _ in range(num_detection_layers):
            self.detection_self_attn_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            
            self.detection_cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            
            self.detection_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
        
        # Language attention for detection (similar to RLA in segmentation)
        self.detection_lang_attn = CrossAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        
        # Language projection
        self.lang_proj_det = Linear(768, hidden_dim, False)  # BERT hidden size to model hidden size
        self.lang_weight_det = nn.Parameter(torch.tensor(0.0))
        
        # Advanced classification head with hierarchical structure
        self.class_prediction_layers = nn.ModuleList()
        for i in range(num_detection_layers):
            self.class_prediction_layers.append(
                MLP(hidden_dim, hidden_dim, num_classes, 3)
            )
        
        # Advanced bbox regression head with hierarchical structure  
        self.bbox_prediction_layers = nn.ModuleList()
        for i in range(num_detection_layers):
            self.bbox_prediction_layers.append(
                MLP(hidden_dim, bbox_embed_dim, 4, 3)
            )
        
        # Detection confidence head (similar to nt_embed in segmentation)
        self.detection_confidence = MLP(hidden_dim, hidden_dim, 1, 2)
        
        # Feature fusion and refinement
        self.feature_fusion = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Scale and offset parameters for bbox regression
        self.bbox_scale = nn.Parameter(torch.ones(4))
        self.bbox_offset = nn.Parameter(torch.zeros(4))
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters similar to segmentation head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for language projection
        nn.init.zeros_(self.lang_proj_det.weight)
        
        # Initialize bbox scale to reasonable values
        nn.init.constant_(self.bbox_scale, 1.0)
        nn.init.constant_(self.bbox_offset, 0.0)
    
    def forward(
        self, 
        shared_features: torch.Tensor,  # From segmentation transformer
        lang_feat: torch.Tensor,        # Language features
        mask_features: torch.Tensor,    # Mask features for spatial attention
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            shared_features: (seq_len, batch, hidden_dim) - shared query features from segmentation
            lang_feat: (batch, 768, seq_len) - language features from BERT
            mask_features: (batch, mask_dim, H, W) - mask features for spatial grounding
        
        Returns:
            Dictionary with detection predictions at multiple layers
        """
        batch_size = shared_features.shape[1]
        
        # Initialize detection queries
        detection_queries = self.detection_queries.weight.unsqueeze(1).repeat(1, batch_size, 1)
        detection_pos = self.detection_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Fuse shared features with detection-specific queries
        fused_features, _ = self.feature_fusion(
            detection_queries, shared_features, shared_features
        )
        fused_features = self.fusion_norm(fused_features + detection_queries)
        
        # Store predictions from each layer
        all_class_predictions = []
        all_bbox_predictions = []
        
        current_features = fused_features
        
        # Multi-layer detection transformer
        for layer_idx in range(self.num_detection_layers):
            # Self attention among detection queries
            current_features = self.detection_self_attn_layers[layer_idx](
                current_features,
                query_pos=detection_pos
            )
            
            # Cross attention with shared visual features
            current_features = self.detection_cross_attn_layers[layer_idx](
                current_features,
                shared_features,
                query_pos=detection_pos,
                pos=None
            )
            
            # Language attention (Detection-specific RLA)
            if layer_idx == 0:  # Apply language attention in first layer
                lang_feat_proj = self.lang_proj_det(lang_feat.permute(0, 2, 1))  # (B, seq_len, hidden_dim)
                lang_attention = self.detection_lang_attn(
                    current_features, 
                    lang_feat_proj.permute(1, 0, 2)  # (seq_len, batch, hidden_dim)
                )
                lang_attention = lang_attention * F.sigmoid(self.lang_weight_det)
                current_features = current_features + lang_attention * 0.1  # Similar to segmentation RLA weight
            
            # FFN layer
            current_features = self.detection_ffn_layers[layer_idx](current_features)
            
            # Generate predictions for this layer
            layer_features = current_features.transpose(0, 1)  # (batch, seq_len, hidden_dim)
            
            # Classification prediction
            class_pred = self.class_prediction_layers[layer_idx](layer_features)
            all_class_predictions.append(class_pred)
            
            # Bbox regression prediction with scale and offset
            bbox_pred = self.bbox_prediction_layers[layer_idx](layer_features)
            bbox_pred = bbox_pred * self.bbox_scale + self.bbox_offset
            all_bbox_predictions.append(bbox_pred)
        
        # Detection confidence (global feature)
        final_features = current_features.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        detection_conf = self.detection_confidence(final_features.mean(dim=1))  # Global average pooling
        
        return {
            "pred_logits": all_class_predictions[-1],  # Final layer prediction
            "pred_boxes": all_bbox_predictions[-1],    # Final layer prediction
            "all_class_predictions": all_class_predictions,  # All layer predictions
            "all_bbox_predictions": all_bbox_predictions,    # All layer predictions
            "detection_confidence": detection_conf,           # Overall detection confidence
            "detection_features": final_features,             # Final detection features
        }


class AdvancedDetectionHead(nn.Module):
    """
    Advanced Detection Head with transformer decoder and sophisticated prediction mechanisms.
    Matches the complexity of the segmentation head.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_detection_layers: int = 3,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        bbox_loss_type: str = "l1",  # "l1", "giou", "diou"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.bbox_loss_type = bbox_loss_type
        
        # Input projection if needed
        self.input_proj = None
        if input_dim != hidden_dim:
            self.input_proj = Linear(input_dim, hidden_dim)
        
        # Core detection transformer decoder
        self.detection_decoder = DetectionTransformerDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_detection_layers=num_detection_layers,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Additional spatial reasoning layers
        self.spatial_reasoning = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, nheads // 2, dropout=dropout)
            for _ in range(2)
        ])
        
        # Hierarchical classification (coarse to fine)
        self.coarse_classifier = MLP(hidden_dim, hidden_dim, 2, 2)  # Person/Background
        self.fine_classifier = MLP(hidden_dim, hidden_dim, num_classes, 3)  # Detailed classes
        
        # Multi-scale bbox regression
        self.bbox_scales = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 2) for _ in range(3)  # Different scales
        ])
        
        # Bbox quality estimation
        self.bbox_quality = MLP(hidden_dim, hidden_dim, 1, 2)
        
        # Feature enhancement
        self.feature_enhancer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        query_features: torch.Tensor,
        lang_feat: Optional[torch.Tensor] = None,
        mask_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: (batch, num_queries, input_dim) - Query features from shared transformer
            lang_feat: (batch, 768, seq_len) - Language features
            mask_features: (batch, mask_dim, H, W) - Mask features
        
        Returns:
            Dictionary with comprehensive detection outputs
        """
        batch_size, num_queries, _ = query_features.shape
        
        # Project input features if needed
        if self.input_proj is not None:
            query_features = self.input_proj(query_features)
        
        # Enhance features
        enhanced_features = self.feature_enhancer(query_features)
        
        # Prepare for transformer decoder (seq_len, batch, hidden_dim)
        transformer_input = enhanced_features.transpose(0, 1)
        
        # Apply detection transformer decoder
        if lang_feat is not None and mask_features is not None:
            detection_outputs = self.detection_decoder(
                transformer_input, lang_feat, mask_features
            )
        else:
            # Fallback for testing without language/mask features
            detection_outputs = self._simple_forward(enhanced_features)
        
        # Additional spatial reasoning
        spatial_features = enhanced_features.transpose(0, 1)  # (num_queries, batch, hidden_dim)
        for spatial_layer in self.spatial_reasoning:
            spatial_features, _ = spatial_layer(spatial_features, spatial_features, spatial_features)
        spatial_features = spatial_features.transpose(0, 1)  # Back to (batch, num_queries, hidden_dim)
        
        # Hierarchical classification
        coarse_logits = self.coarse_classifier(spatial_features)
        fine_logits = self.fine_classifier(spatial_features)
        
        # Multi-scale bbox regression
        bbox_predictions = []
        for bbox_head in self.bbox_scales:
            bbox_pred = bbox_head(spatial_features)
            bbox_predictions.append(bbox_pred)
        
        # Combine bbox predictions (ensemble)
        final_bbox = torch.stack(bbox_predictions, dim=-1).mean(dim=-1)
        
        # Bbox quality estimation
        bbox_quality = self.bbox_quality(spatial_features)
        
        # Combine all outputs
        final_outputs = {
            **detection_outputs,
            "hierarchical_coarse": coarse_logits,
            "hierarchical_fine": fine_logits,
            "multi_scale_boxes": bbox_predictions,
            "final_boxes": final_bbox,
            "bbox_quality": bbox_quality,
            "enhanced_features": enhanced_features,
        }
        
        return final_outputs
    
    def _simple_forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified forward for testing without language features."""
        # Basic classification and bbox regression
        class_logits = self.fine_classifier(features)
        bbox_pred = self.bbox_scales[0](features)
        
        return {
            "pred_logits": class_logits,
            "pred_boxes": bbox_pred,
            "detection_confidence": torch.zeros(features.shape[0], 1, device=features.device),
            "detection_features": features,
        }


def build_advanced_detection_head(cfg, input_dim: int):
    """Build advanced detection head from configuration."""
    return AdvancedDetectionHead(
        input_dim=input_dim,
        num_classes=cfg.get("NUM_DETECTION_CLASSES", 1),
        hidden_dim=cfg.get("DETECTION_HIDDEN_DIM", 256),
        num_detection_layers=cfg.get("NUM_DETECTION_LAYERS", 3),
        nheads=cfg.get("DETECTION_NHEADS", 8),
        dim_feedforward=cfg.get("DETECTION_DIM_FEEDFORWARD", 2048),
        dropout=cfg.get("DETECTION_DROPOUT", 0.1),
        bbox_loss_type=cfg.get("BBOX_LOSS_TYPE", "l1"),
    )