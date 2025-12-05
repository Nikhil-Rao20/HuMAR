# ------------------------------------------------------------------------
# PoolFormer Backbone
# MetaFormer baseline using simple pooling
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import timm
from typing import List


class PoolFormerBackbone(nn.Module):
    """
    PoolFormer Backbone wrapper for UniPHD
    MetaFormer architecture using pooling instead of attention
    """
    
    def __init__(self, model_name='poolformer_s12', pretrained=True, out_indices=(0, 1, 2, 3)):
        """
        Args:
            model_name: 'poolformer_s12', 'poolformer_s24', 'poolformer_s36'
            pretrained: Load pretrained weights from timm
            out_indices: Which stages to output (0-3 for 4 stages)
        """
        super().__init__()
        
        self.model_name = model_name
        self.out_indices = out_indices
        
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )
        except Exception as e:
            print(f"Warning: Could not load {model_name} from timm. Error: {e}")
            print("Falling back to poolformer_s12")
            self.backbone = timm.create_model(
                'poolformer_s12',
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )
        
        # Get number of output channels for each stage
        self.num_features = self.backbone.feature_info.channels()
        
        print(f"Loaded {model_name} backbone")
        print(f"Output channels per stage: {self.num_features}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            List of feature maps from selected stages
        """
        features = self.backbone(x)
        return features


def build_poolformer_backbone(model_name='poolformer_s12', pretrained=True, out_indices=(0, 1, 2, 3), **kwargs):
    """
    Build PoolFormer backbone
    
    Args:
        model_name: 'poolformer_s12', 'poolformer_s24', or 'poolformer_s36'
        pretrained: Whether to load pretrained weights
        out_indices: Which stages to output
    
    Returns:
        PoolFormerBackbone model
    """
    model = PoolFormerBackbone(
        model_name=model_name,
        pretrained=pretrained,
        out_indices=out_indices
    )
    return model


# Model specifications
POOLFORMER_SPECS = {
    'poolformer_s12': {
        'params': '12M',
        'channels': [64, 128, 320, 512],
        'description': 'Small PoolFormer with 12 layers'
    },
    'poolformer_s24': {
        'params': '21M',
        'channels': [64, 128, 320, 512],
        'description': 'Medium PoolFormer with 24 layers'
    },
    'poolformer_s36': {
        'params': '31M',
        'channels': [64, 128, 320, 512],
        'description': 'Large PoolFormer with 36 layers'
    }
}
