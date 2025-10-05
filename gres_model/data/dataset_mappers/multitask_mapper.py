# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))
import copy
import logging
import numpy as np
import torch
from typing import List, Union
import cv2

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)

"""
Custom data mapper for multitask learning: detection, segmentation, and pose estimation.
"""

logger = logging.getLogger(__name__)

class MultitaskDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by multitask models.

    This mapper handles detection, segmentation, and keypoint annotations simultaneously.
    """

    def __init__(self, cfg, is_train=True):
        """
        Args:
            cfg: config object
            is_train: whether we're in training mode
        """
        # Disable crop augmentations
        self.crop_gen = None

        # Disable most transform augmentations to avoid flip issues
        self.tfm_gens = []  # No augmentations
        
        # Multitask specific settings
        self.enable_detection = cfg.MODEL.get("ENABLE_DETECTION", True)
        self.enable_segmentation = cfg.MODEL.get("ENABLE_SEGMENTATION", True) 
        self.enable_keypoints = cfg.MODEL.get("ENABLE_KEYPOINTS", True)

        # Format settings
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON if hasattr(cfg.MODEL, 'MASK_ON') else self.enable_segmentation
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON if hasattr(cfg.MODEL, 'KEYPOINT_ON') else self.enable_keypoints
        
        self.is_train = is_train
        self.cfg = cfg
    
    def _get_keypoint_flip_indices(self):
        """
        Get keypoint flip indices for COCO format keypoints.
        Returns the indices for flipping left/right keypoints during data augmentation.
        """
        # COCO keypoint flip map for horizontal flipping
        # Format: [left_keypoint_idx, right_keypoint_idx]
        flip_map = [
            [1, 2],   # left_eye <-> right_eye
            [3, 4],   # left_ear <-> right_ear
            [5, 6],   # left_shoulder <-> right_shoulder
            [7, 8],   # left_elbow <-> right_elbow
            [9, 10],  # left_wrist <-> right_wrist
            [11, 12], # left_hip <-> right_hip
            [13, 14], # left_knee <-> right_knee
            [15, 16], # left_ankle <-> right_ankle
        ]
        
        # Create flip indices array
        flip_indices = list(range(17))  # Initialize with identity mapping
        for left_idx, right_idx in flip_map:
            flip_indices[left_idx] = right_idx
            flip_indices[right_idx] = left_idx
        
        return flip_indices

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # Load image
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Apply augmentations
        if self.crop_gen is not None:
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.crop_gen.get_crop_size(image.shape[:2]),
                image.shape[:2],
                np.random.choice(dataset_dict["annotations"]) if dataset_dict["annotations"] else None,
            )
            image = crop_tfm.apply_image(image)
        
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        if self.crop_gen is not None:
            transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Process annotations
        if "annotations" not in dataset_dict:
            return dataset_dict

        # Apply transforms to annotations (disable keypoint flipping)
        annos = [
            utils.transform_instance_annotations(
                obj,
                transforms, 
                image_shape,
                keypoint_hflip_indices=None,  # Disable keypoint flipping to avoid errors
            )
            for obj in dataset_dict.pop("annotations")
        ]

        # Filter out annotations
        annos = [obj for obj in annos if obj.get("iscrowd", 0) == 0]

        # Create instances
        instances = utils.annotations_to_instances(
            annos, 
            image_shape,
            mask_format=self.cfg.INPUT.MASK_FORMAT if hasattr(self.cfg.INPUT, 'MASK_FORMAT') else 'polygon'
        )

        # Handle keypoints specifically
        if self.keypoint_on and len(annos) > 0:
            keypoints = [obj.get("keypoints", []) for obj in annos]
            keypoints = utils.filter_empty_instances(
                utils.create_keypoint_instances(keypoints, image_shape),
                by_box=self.enable_detection,
                by_mask=self.enable_segmentation,
            )
            instances.gt_keypoints = keypoints

        # Create separate target dictionaries for each task
        targets = {}
        
        if self.enable_detection:
            targets["detection"] = {
                "boxes": instances.gt_boxes if hasattr(instances, 'gt_boxes') else Boxes.cat([]),
                "classes": instances.gt_classes if hasattr(instances, 'gt_classes') else torch.tensor([]),
            }
        
        if self.enable_segmentation and hasattr(instances, 'gt_masks'):
            targets["segmentation"] = {
                "masks": instances.gt_masks,
                "classes": instances.gt_classes if hasattr(instances, 'gt_classes') else torch.tensor([]),
            }
        
        if self.enable_keypoints and hasattr(instances, 'gt_keypoints'):
            targets["keypoints"] = {
                "keypoints": instances.gt_keypoints,
                "classes": instances.gt_classes if hasattr(instances, 'gt_classes') else torch.tensor([]),
            }

        dataset_dict["instances"] = instances
        dataset_dict["targets"] = targets
        
        # Add referring expression information if available
        if "sentences" in dataset_dict:
            dataset_dict["text"] = [sent["sent"] for sent in dataset_dict["sentences"]]
        else:
            # Provide default text if no referring expression
            dataset_dict["text"] = ["person"]  # Default referring expression
        
        # Add language tokens for the model
        # Use the first sentence as the referring expression
        text = dataset_dict["text"][0] if dataset_dict["text"] else "person"
        
        # Simple tokenization - split by spaces and convert to dummy token IDs
        # For a proper implementation, you'd use BERT tokenizer here
        tokens = text.lower().split()
        
        # Create dummy token IDs (in a real implementation, use proper BERT tokenizer)
        # For now, create a simple mapping
        token_to_id = {
            'person': 1, 'man': 2, 'woman': 3, 'child': 4, 'boy': 5, 'girl': 6,
            'standing': 7, 'sitting': 8, 'walking': 9, 'running': 10,
            'wearing': 11, 'holding': 12, 'carrying': 13,
            'left': 14, 'right': 15, 'center': 16, 'front': 17, 'back': 18,
            'red': 19, 'blue': 20, 'green': 21, 'black': 22, 'white': 23,
            'shirt': 24, 'pants': 25, 'dress': 26, 'hat': 27, 'shoes': 28
        }
        
        # Convert tokens to IDs (default to 0 for unknown tokens)
        token_ids = [token_to_id.get(token, 0) for token in tokens]
        
        # Pad or truncate to fixed length (20 tokens as in config)
        max_tokens = 20
        if len(token_ids) < max_tokens:
            token_ids.extend([0] * (max_tokens - len(token_ids)))  # Pad with 0
        else:
            token_ids = token_ids[:max_tokens]  # Truncate
        
        dataset_dict["lang_tokens"] = torch.tensor(token_ids, dtype=torch.long)
        
        # Create attention mask for language tokens (1 for real tokens, 0 for padding)
        lang_mask = [1 if token_id > 0 else 0 for token_id in token_ids]
        dataset_dict["lang_mask"] = torch.tensor(lang_mask, dtype=torch.bool)
        
        return dataset_dict


def create_keypoint_instances(keypoints_list, image_shape):
    """
    Create Keypoints instances from a list of keypoint annotations.
    
    Args:
        keypoints_list: list of keypoint annotations, each in format [x1,y1,v1,...,x17,y17,v17]
        image_shape: (H, W) of the image
        
    Returns:
        Keypoints object
    """
    if not keypoints_list:
        return Keypoints(torch.zeros((0, 17, 3)))
    
    keypoints_tensor = []
    for kpts in keypoints_list:
        if len(kpts) == 51:  # 17 keypoints * 3 (x, y, visibility)
            # Reshape from flat list to (17, 3)
            kpts_array = np.array(kpts).reshape(-1, 3)
            keypoints_tensor.append(kpts_array)
        else:
            # Handle invalid keypoints
            logger.warning(f"Invalid keypoints format: expected 51 values, got {len(kpts)}")
            keypoints_tensor.append(np.zeros((17, 3)))
    
    return Keypoints(torch.as_tensor(np.stack(keypoints_tensor)))


def multitask_collate_fn(batch):
    """
    Custom collate function for multitask learning.
    Handles variable number of instances per image for different tasks.
    """
    # Group by common keys
    batched = {}
    
    # Handle images (always present)
    batched["image"] = torch.stack([item["image"] for item in batch])
    
    # Handle instances and targets
    batched["instances"] = [item["instances"] for item in batch if "instances" in item]
    batched["targets"] = [item["targets"] for item in batch if "targets" in item]
    
    # Handle text (referring expressions)
    if "text" in batch[0]:
        batched["text"] = [item["text"] for item in batch]
    
    # Handle other metadata
    for key in ["file_name", "image_id", "height", "width"]:
        if key in batch[0]:
            batched[key] = [item[key] for item in batch]
    
    return batched