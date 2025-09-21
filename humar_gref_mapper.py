#!/usr/bin/env python3
"""
HuMAR-GREF dataset mapper for ReLA training.
"""

import copy
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add local detectron2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, Boxes

class HumarGrefDatasetMapper:
    """
    Dataset mapper for HuMAR-GREF referring expression dataset.
    """
    
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.use_keypoint = getattr(cfg.MODEL, 'KEYPOINT_ON', False)
        
        # Build augmentations
        augs = []
        if is_train:
            augs.append(T.RandomFlip())
        augs.append(T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN], 
            cfg.INPUT.MAX_SIZE_TRAIN
        ))
        
        self.tfm_gens = augs
        self.img_format = cfg.INPUT.FORMAT
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # Read image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # Apply transforms
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.tfm_gens)(aug_input)
        image = aug_input.image
        
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        # Process annotations
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            
            # Create instances
            instances = utils.annotations_to_instances(annos, image_shape)
            
            # Add referring expression data
            if len(annos) > 0:
                # Get the first annotation's referring expression data
                # (In practice, you might want to handle multiple expressions differently)
                first_anno = annos[0]
                if "sentences" in first_anno:
                    instances.sentences = first_anno["sentences"]
                if "ref_id" in first_anno:
                    instances.ref_id = first_anno["ref_id"]
            
            dataset_dict["instances"] = instances
            
        return dataset_dict


def build_humar_gref_train_loader(cfg):
    """
    Build a dataloader for HuMAR-GREF dataset.
    """
    from detectron2.data import build_detection_train_loader
    from detectron2.data.common import DatasetFromList, MapDataset
    from detectron2.data.dataset_mapper import DatasetMapper
    from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
    from detectron2.data.build import get_detection_dataset_dicts
    from torch.utils.data import DataLoader
    
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
    )
    
    dataset = DatasetFromList(dataset_dicts, copy=False)
    
    mapper = HumarGrefDatasetMapper(cfg, is_train=True)
    dataset = MapDataset(dataset, mapper)
    
    sampler = TrainingSampler(len(dataset))
    
    return DataLoader(
        dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=utils.collate_fn,
    )