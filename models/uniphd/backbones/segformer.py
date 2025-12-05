# ------------------------------------------------------------------------
# SegFormer MiT (Mix Transformer) Backbone
# Supports MiT-B0 and MiT-B1 from Hugging Face transformers
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import SegformerModel
from typing import List


class SegFormerBackbone(nn.Module):
    """
    SegFormer MiT Backbone wrapper for UniPHD using Hugging Face
    Supports MiT-B0 (3.7M params) and MiT-B1 (13.7M params)
    """
    
    def __init__(self, model_name='nvidia/mit-b0', pretrained=True, out_indices=(0, 1, 2, 3)):
        """
        Args:
            model_name: 'nvidia/mit-b0' or 'nvidia/mit-b1' from Hugging Face
            pretrained: Load pretrained weights from Hugging Face
            out_indices: Which stages to output (0-3 for 4 stages)
        """
        super().__init__()
        
        self.model_name = model_name
        self.out_indices = out_indices
        
        # Load SegFormer from Hugging Face
        if pretrained:
            self.backbone = SegformerModel.from_pretrained(model_name)
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_name)
            self.backbone = SegformerModel(config)
        
        # Enable output of hidden states
        self.backbone.config.output_hidden_states = True
        
        # Define channel dimensions for each variant
        # MiT-B0: [32, 64, 160, 256]
        # MiT-B1: [64, 128, 320, 512]
        if 'b0' in model_name.lower():
            all_channels = [32, 64, 160, 256]
        elif 'b1' in model_name.lower():
            all_channels = [64, 128, 320, 512]
        else:
            all_channels = [32, 64, 160, 256]  # Default to B0
        
        # Store the number of features for each selected stage
        self.num_features = [all_channels[i] for i in out_indices]
        
        print(f"Loaded SegFormer {model_name} backbone from Hugging Face")
        print(f"Output channels per stage: {self.num_features}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            List of feature maps from selected stages
        """
        # Get outputs from SegFormer encoder
        outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        
        # Extract hidden states from encoder
        # hidden_states: (patch_embeddings, stage_1, stage_2, stage_3, stage_4)
        hidden_states = outputs.hidden_states
        
        # Convert from (B, H*W, C) to (B, C, H, W) format
        features = []
        for idx in self.out_indices:
            # Get feature at this stage (add 1 to skip patch embeddings)
            feat = hidden_states[idx + 1]
            B, N, C = feat.shape
            
            # Calculate spatial dimensions (H, W) from N
            H = W = int(N ** 0.5)
            
            # Reshape: (B, H*W, C) -> (B, C, H, W)
            feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
            features.append(feat)
        
        return features


def build_segformer_backbone(model_name='segformer_mit_b0', pretrained=True, out_indices=(0, 1, 2, 3), **kwargs):
    """
    Build SegFormer backbone from Hugging Face
    
    Args:
        model_name: 'segformer_mit_b0' or 'segformer_mit_b1'
        pretrained: Whether to load pretrained weights
        out_indices: Which stages to output (0-3)
    
    Returns:
        SegFormerBackbone model
    """
    # Map our naming to Hugging Face model IDs
    hf_model_mapping = {
        'segformer_mit_b0': 'nvidia/mit-b0',
        'segformer_mit_b1': 'nvidia/mit-b1',
    }
    
    hf_model_name = hf_model_mapping.get(model_name, 'nvidia/mit-b0')
    
    model = SegFormerBackbone(
        model_name=hf_model_name,
        pretrained=pretrained,
        out_indices=out_indices
    )
    return model


# Model specifications (updated for correct naming)
SEGFORMER_SPECS = {
    'segformer_mit_b0': {
        'params': '3.7M',
        'channels': [32, 64, 160, 256],  # C1, C2, C3, C4
        'description': 'Lightweight SegFormer MiT-B0 backbone from Hugging Face'
    },
    'segformer_mit_b1': {
        'params': '13.7M',
        'channels': [64, 128, 320, 512],
        'description': 'Balanced SegFormer MiT-B1 backbone from Hugging Face'
    }
}
