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
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        
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

        # Apply transforms to annotations
        annos = [
            utils.transform_instance_annotations(
                obj,
                transforms, 
                image_shape,
                keypoint_hflip_indices=utils.create_keypoint_hflip_indices(
                    MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).keypoint_names
                ) if self.keypoint_on else None,
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