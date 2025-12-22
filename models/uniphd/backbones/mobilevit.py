# ------------------------------------------------------------------------
# MobileViT Backbone
# Lightweight vision transformer for mobile devices
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict
from util.misc import NestedTensor


class MobileViTBackbone(nn.Module):
    """
    MobileViT Backbone wrapper for UniPHD
    Efficient vision transformer for resource-constrained environments
    """
    
    def __init__(self, model_name='mobilevit_s', pretrained=True, out_indices=(0, 1, 2, 3)):
        """
        Args:
            model_name: 'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s'
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
            print("Falling back to mobilevit_s")
            self.backbone = timm.create_model(
                'mobilevit_s',
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )
        
        # Get number of output channels for each stage
        self.num_features = self.backbone.feature_info.channels()
        
        print(f"Loaded {model_name} backbone")
        print(f"Output channels per stage: {self.num_features}")
    
    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor containing tensors and mask
        Returns:
            Dict of NestedTensor feature maps from selected stages
        """
        # Extract the actual tensor from NestedTensor
        x = tensor_list.tensors
        mask = tensor_list.mask
        
        # Forward through backbone
        features = self.backbone(x)
        
        # Convert features to NestedTensor format with interpolated masks
        out: Dict[str, NestedTensor] = {}
        for idx, feat in enumerate(features):
            # Interpolate mask to match feature size
            m = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[str(idx)] = NestedTensor(feat, m)
        
        return out


def build_mobilevit_backbone(model_name='mobilevit_s', pretrained=True, out_indices=(0, 1, 2, 3), **kwargs):
    """
    Build MobileViT backbone
    
    Args:
        model_name: 'mobilevit_xxs', 'mobilevit_xs', or 'mobilevit_s'
        pretrained: Whether to load pretrained weights
        out_indices: Which stages to output
    
    Returns:
        MobileViTBackbone model
    """
    model = MobileViTBackbone(
        model_name=model_name,
        pretrained=pretrained,
        out_indices=out_indices
    )
    return model


# Model specifications
MOBILEVIT_SPECS = {
    'mobilevit_xxs': {
        'params': '1.3M',
        'description': 'Extra-extra-small MobileViT for edge devices'
    },
    'mobilevit_xs': {
        'params': '2.3M',
        'description': 'Extra-small MobileViT'
    },
    'mobilevit_s': {
        'params': '5.6M',
        'description': 'Small MobileViT (recommended)'
    }
}
