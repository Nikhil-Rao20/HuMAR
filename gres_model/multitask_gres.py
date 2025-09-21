"""
Multitask ReLA (GRES) model for detection, segmentation, and pose estimation.
Extends the original GRES model with additional heads for detection and keypoint estimation.
"""

# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))
from typing import Tuple, Dict, List
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances, BitMasks, Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import Linear, ShapeSpec

from .modeling.criterion import ReferringCriterion


# Import advanced heads
try:
    from .detection_head import AdvancedDetectionHead, build_advanced_detection_head
    from .pose_head import AdvancedPoseHead, build_advanced_pose_head
    ADVANCED_HEADS_AVAILABLE = True
except ImportError:
    ADVANCED_HEADS_AVAILABLE = False
    
    # Fallback to simple heads if advanced ones are not available
    class DetectionHead(nn.Module):
        """Simple detection head for fallback."""
        def __init__(self, input_dim: int, num_classes: int = 1, hidden_dim: int = 256):
            super().__init__()
            self.num_classes = num_classes
            
            # Classification head
            self.class_head = nn.Sequential(
                Linear(input_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, num_classes)
            )
            
            # Box regression head  
            self.bbox_head = nn.Sequential(
                Linear(input_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, 4)
            )

        def forward(self, x):
            class_logits = self.class_head(x)
            bbox_pred = self.bbox_head(x)
            return {"pred_logits": class_logits, "pred_boxes": bbox_pred}

    class KeypointHead(nn.Module):
        """Simple keypoint head for fallback."""
        def __init__(self, input_dim: int, num_keypoints: int = 17, hidden_dim: int = 256):
            super().__init__()
            self.num_keypoints = num_keypoints
            
            self.keypoint_head = nn.Sequential(
                Linear(input_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, num_keypoints * 3)
            )

        def forward(self, x):
            keypoint_pred = self.keypoint_head(x)
            N = x.shape[0]
            keypoint_pred = keypoint_pred.view(N, self.num_keypoints, 3)
            return {"pred_keypoints": keypoint_pred}


@META_ARCH_REGISTRY.register()
class MultitaskGRES(nn.Module):
    """
    Multitask GRES model that handles referring expression segmentation,
    detection, and pose estimation simultaneously.
    """
    
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        lang_backbone: nn.Module,
        # multitask specific
        enable_detection: bool = True,
        enable_keypoints: bool = True,
        detection_head: nn.Module = None,
        keypoint_head: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # language backbone
        self.text_encoder = lang_backbone
        
        # multitask heads
        self.enable_detection = enable_detection
        self.enable_keypoints = enable_keypoints
        self.detection_head = detection_head
        self.keypoint_head = keypoint_head

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        text_encoder = BertModel.from_pretrained(cfg.REFERRING.BERT_TYPE)
        text_encoder.pooler = None

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        losses = ["masks"]

        criterion = ReferringCriterion(
            weight_dict=weight_dict,
            losses=losses,
        )
        
        # Multitask configuration
        enable_detection = cfg.MODEL.get("ENABLE_DETECTION", True)
        enable_keypoints = cfg.MODEL.get("ENABLE_KEYPOINTS", True)
        
        # Get feature dimension from transformer decoder
        transformer_hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        
        # Create task-specific heads
        detection_head = None
        keypoint_head = None
        
        if enable_detection:
            if ADVANCED_HEADS_AVAILABLE:
                detection_head = AdvancedDetectionHead(
                    input_dim=transformer_hidden_dim,
                    num_classes=1,  # Only person class
                    hidden_dim=cfg.MODEL.get("DETECTION_HEAD_DIM", 256),
                    num_detection_layers=cfg.MODEL.get("NUM_DETECTION_LAYERS", 3),
                    nheads=cfg.MODEL.get("DETECTION_NHEADS", 8),
                    dim_feedforward=cfg.MODEL.get("DETECTION_DIM_FEEDFORWARD", 2048),
                )
            else:
                detection_head = DetectionHead(
                    input_dim=transformer_hidden_dim,
                    num_classes=1,  # Only person class
                    hidden_dim=cfg.MODEL.get("DETECTION_HEAD_DIM", 256)
                )
        
        if enable_keypoints:
            if ADVANCED_HEADS_AVAILABLE:
                keypoint_head = AdvancedPoseHead(
                    input_dim=transformer_hidden_dim,
                    num_keypoints=17,  # COCO keypoints
                    hidden_dim=cfg.MODEL.get("KEYPOINT_HEAD_DIM", 256),
                    num_pose_layers=cfg.MODEL.get("NUM_POSE_LAYERS", 4),
                    nheads=cfg.MODEL.get("POSE_NHEADS", 8),
                    dim_feedforward=cfg.MODEL.get("POSE_DIM_FEEDFORWARD", 2048),
                )
            else:
                keypoint_head = KeypointHead(
                    input_dim=transformer_hidden_dim,
                    num_keypoints=17,  # COCO keypoints
                    hidden_dim=cfg.MODEL.get("KEYPOINT_HEAD_DIM", 256)
                )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "lang_backbone": text_encoder,
            # multitask
            "enable_detection": enable_detection,
            "enable_keypoints": enable_keypoints,
            "detection_head": detection_head,
            "keypoint_head": keypoint_head,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        lang_emb = [x['lang_tokens'].to(self.device) for x in batched_inputs]
        lang_emb = torch.cat(lang_emb, dim=0)

        lang_mask = [x['lang_mask'].to(self.device) for x in batched_inputs]
        lang_mask = torch.cat(lang_mask, dim=0)

        lang_feat = self.text_encoder(lang_emb, attention_mask=lang_mask)[0] # B, Nl, 768

        lang_feat = lang_feat.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        features = self.backbone(images.tensor, lang_feat, lang_mask)
        outputs = self.sem_seg_head(features, lang_feat, lang_mask)

        # Extract query features for additional heads
        # Assuming the transformer decoder outputs include query features
        if "pred_logits" in outputs or "query_features" in outputs:
            # Get query features from transformer decoder
            query_features = outputs.get("query_features", None)
            if query_features is None and "pred_logits" in outputs:
                # If query_features not available, we need to modify the transformer decoder
                # For now, we'll use a placeholder
                batch_size, num_queries = outputs["pred_logits"].shape[:2]
                hidden_dim = outputs["pred_logits"].shape[-1] if len(outputs["pred_logits"].shape) > 2 else 256
                query_features = torch.zeros(batch_size, num_queries, hidden_dim, device=self.device)
            
            # Apply additional heads
            if self.enable_detection and self.detection_head is not None:
                # Flatten query features for detection head
                batch_size, num_queries, hidden_dim = query_features.shape
                flat_queries = query_features.view(-1, hidden_dim)
                detection_outputs = self.detection_head(flat_queries)
                
                # Reshape back to batch format
                for key, value in detection_outputs.items():
                    if len(value.shape) == 2:
                        outputs[key] = value.view(batch_size, num_queries, -1)
                    else:
                        outputs[key] = value.view(batch_size, num_queries, *value.shape[1:])
            
            if self.enable_keypoints and self.keypoint_head is not None:
                # Flatten query features for keypoint head
                batch_size, num_queries, hidden_dim = query_features.shape
                flat_queries = query_features.view(-1, hidden_dim)
                keypoint_outputs = self.keypoint_head(flat_queries)
                
                # Reshape back to batch format
                for key, value in keypoint_outputs.items():
                    if len(value.shape) == 3:  # (N, num_keypoints, 3)
                        outputs[key] = value.view(batch_size, num_queries, *value.shape[1:])
                    else:
                        outputs[key] = value.view(batch_size, num_queries, *value.shape[1:])

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            losses = self.criterion(outputs, targets)

            # Add multitask losses here if needed
            # TODO: Implement detection and keypoint losses

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            return self.inference(outputs, batched_inputs, images)

    def inference(self, outputs, batched_inputs, images):
        """Handle inference for all tasks."""
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        nt_pred_results = outputs["nt_label"]

        processed_results = []
        for i, (mask_pred_result, nt_pred_result, input_per_image, image_size) in enumerate(zip(
            mask_pred_results, nt_pred_results, batched_inputs, images.image_sizes
        )):
            result = {}
            
            # Segmentation results
            r, nt = retry_if_cuda_oom(self.refer_inference)(mask_pred_result, nt_pred_result)
            result["ref_seg"] = r
            result["nt_label"] = nt
            
            # Detection results
            if self.enable_detection and "pred_logits" in outputs and "pred_boxes" in outputs:
                det_logits = outputs["pred_logits"][i]  # (num_queries, num_classes)
                det_boxes = outputs["pred_boxes"][i]    # (num_queries, 4)
                
                # Apply detection post-processing
                det_results = self.detection_inference(det_logits, det_boxes, image_size)
                result["detection"] = det_results
            
            # Keypoint results
            if self.enable_keypoints and "pred_keypoints" in outputs:
                kpt_pred = outputs["pred_keypoints"][i]  # (num_queries, 17, 3)
                
                # Apply keypoint post-processing
                kpt_results = self.keypoint_inference(kpt_pred, image_size)
                result["keypoints"] = kpt_results
            
            processed_results.append(result)

        return processed_results

    def detection_inference(self, logits, boxes, image_size):
        """Post-process detection predictions."""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Simple thresholding for now
        scores, labels = probs.max(dim=-1)
        keep = scores > 0.5
        
        final_boxes = boxes[keep]
        final_scores = scores[keep]
        final_labels = labels[keep]
        
        return {
            "boxes": final_boxes,
            "scores": final_scores, 
            "labels": final_labels
        }

    def keypoint_inference(self, keypoints, image_size):
        """Post-process keypoint predictions."""
        # keypoints: (num_queries, 17, 3)
        
        # Apply sigmoid to visibility scores
        keypoints_processed = keypoints.clone()
        keypoints_processed[:, :, 2] = torch.sigmoid(keypoints_processed[:, :, 2])
        
        # Filter by visibility threshold
        vis_threshold = 0.3
        valid_instances = (keypoints_processed[:, :, 2] > vis_threshold).any(dim=1)
        
        return {
            "keypoints": keypoints_processed[valid_instances],
            "num_instances": valid_instances.sum().item()
        }

    def prepare_targets(self, batched_inputs, images):
        """Prepare targets for multitask training."""
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        for data_per_image in batched_inputs:
            # pad instances
            targets_per_image = data_per_image['instances'].to(self.device)
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            is_empty = torch.tensor(data_per_image['empty'], dtype=targets_per_image.gt_classes.dtype, device=targets_per_image.gt_classes.device)
            
            target_dict = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
                "empty": is_empty,
            }
            
            # Add detection targets
            if self.enable_detection and hasattr(targets_per_image, 'gt_boxes'):
                target_dict["boxes"] = targets_per_image.gt_boxes.tensor
            
            # Add keypoint targets  
            if self.enable_keypoints and hasattr(targets_per_image, 'gt_keypoints'):
                target_dict["keypoints"] = targets_per_image.gt_keypoints.tensor
            
            if data_per_image["gt_mask_merged"] is not None:
                target_dict["gt_mask_merged"] = data_per_image["gt_mask_merged"].to(self.device)

            new_targets.append(target_dict)
        return new_targets

    def refer_inference(self, mask_pred, nt_pred):
        """Original referring segmentation inference."""
        mask_pred = mask_pred.sigmoid()
        nt_pred = nt_pred.sigmoid()
        return mask_pred, nt_pred