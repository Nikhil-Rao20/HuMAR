# ------------------------------------------------------------------------
# EfficientFormerV2 Backbone
# Efficient vision transformer with improved performance
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict
from util.misc import NestedTensor


class EfficientFormerV2Backbone(nn.Module):
    """
    EfficientFormerV2 Backbone wrapper for UniPHD
    State-of-the-art efficient vision transformer
    """
    
    def __init__(self, model_name='efficientformerv2_s0', pretrained=True, out_indices=(0, 1, 2, 3)):
        """
        Args:
            model_name: 'efficientformerv2_s0', 'efficientformerv2_s1', 'efficientformerv2_s2'
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
            print("Trying alternative naming...")
            # Try alternative names
            alt_names = ['efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7']
            for alt_name in alt_names:
                try:
                    self.backbone = timm.create_model(
                        alt_name,
                        pretrained=pretrained,
                        features_only=True,
                        out_indices=out_indices
                    )
                    print(f"Successfully loaded {alt_name}")
                    break
                except:
                    continue
        
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


def build_efficientformerv2_backbone(model_name='efficientformerv2_s0', pretrained=True, out_indices=(0, 1, 2, 3), **kwargs):
    """
    Build EfficientFormerV2 backbone
    
    Args:
        model_name: 'efficientformerv2_s0', 'efficientformerv2_s1', or 'efficientformerv2_s2'
        pretrained: Whether to load pretrained weights
        out_indices: Which stages to output
    
    Returns:
        EfficientFormerV2Backbone model
    """
    model = EfficientFormerV2Backbone(
        model_name=model_name,
        pretrained=pretrained,
        out_indices=out_indices
    )
    return model


# Model specifications
EFFICIENTFORMERV2_SPECS = {
    'efficientformerv2_s0': {
        'params': '3.5M',
        'description': 'Small EfficientFormerV2'
    },
    'efficientformerv2_s1': {
        'params': '6.1M',
        'description': 'Medium EfficientFormerV2'
    },
    'efficientformerv2_s2': {
        'params': '12.6M',
        'description': 'Large EfficientFormerV2'
    }
}
