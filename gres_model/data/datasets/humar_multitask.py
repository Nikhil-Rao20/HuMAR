# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))
import contextlib
import io
import logging
import numpy as np
import os
import json
import copy
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, Instances
from detectron2.utils.file_io import PathManager

"""
This file contains functions to parse HuMAR-format annotations into dicts in "Detectron2 format"
for multitask learning: detection, segmentation, and pose estimation.
"""

logger = logging.getLogger(__name__)

__all__ = ["load_humar_multitask_json"]

# COCO keypoint names and skeleton for visualization
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# COCO keypoint skeleton for connecting keypoints
COCO_KEYPOINT_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

def load_humar_multitask_json(
    gref_json_file,
    instances_json_file, 
    image_root,
    split="train",
    extra_annotation_keys=None
):
    """
    Load HuMAR dataset with multitask annotations (detection, segmentation, pose).
    
    Args:
        gref_json_file (str): path to the HuMAR GREF json file
        instances_json_file (str): path to the HuMAR instances json file  
        image_root (str): directory containing images
        split (str): dataset split (train/val/test)
        extra_annotation_keys (list): additional annotation keys to include
        
    Returns:
        list[dict]: dataset records in Detectron2 format
    """
    
    timer = Timer()
    logger.info(f"Loading HuMAR multitask dataset for {split}...")
    
    # Load referring expression data
    logger.info(f"Loading GREF data from {gref_json_file}")
    with open(gref_json_file, 'r') as f:
        gref_data = json.load(f)
    
    # Load instance annotations  
    logger.info(f"Loading instance data from {instances_json_file}")
    with open(instances_json_file, 'r') as f:
        instances_data = json.load(f)
    
    # Create lookup tables
    ann_id_to_instance = {ann['id']: ann for ann in instances_data}
    
    # Filter by split
    gref_data = [item for item in gref_data if item.get('split', 'train') == split]
    
    logger.info(f"Loaded {len(gref_data)} referring expressions and {len(instances_data)} instances")
    
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id", "area", "segmentation", "keypoints"] + (extra_annotation_keys or [])
    
    # Group by image_id for efficiency
    image_groups = {}
    for ref_item in gref_data:
        image_id = ref_item['image_id']
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(ref_item)
    
    logger.info(f"Processing {len(image_groups)} unique images...")
    
    for image_id, ref_items in image_groups.items():
        # Get image info from first referring item
        first_ref = ref_items[0]
        
        record = {}
        record["file_name"] = os.path.join(image_root, first_ref["file_name"])
        record["image_id"] = image_id
        record["source"] = "humar_multitask"
        
        # Get image dimensions (we'll need to load the image for this)
        try:
            with Image.open(record["file_name"]) as img:
                record["width"], record["height"] = img.size
        except Exception as e:
            logger.warning(f"Could not load image {record['file_name']}: {e}")
            continue
        
        # Process all referring expressions for this image
        annotations = []
        
        for ref_item in ref_items:
            for ann_id in ref_item["ann_id"]:
                if ann_id not in ann_id_to_instance:
                    logger.warning(f"Instance {ann_id} not found in instances data")
                    continue
                
                instance = ann_id_to_instance[ann_id]
                
                # Create annotation object
                obj = {key: instance[key] for key in ann_keys if key in instance}
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                
                # Add referring expression information
                obj["ref_id"] = ref_item["ref_id"]
                obj["sentences"] = ref_item["sentences"]
                
                # Process segmentation
                segm = instance.get("segmentation", None)
                if segm:
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, record["height"], record["width"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            logger.warning(f"Invalid segmentation for ann_id {ann_id}")
                            continue
                    obj["segmentation"] = segm
                
                # Process keypoints
                keypoints = instance.get("keypoints", None)
                if keypoints:
                    # Convert keypoints format from [[x,y,v], ...] to [x1,y1,v1,x2,y2,v2,...]
                    if isinstance(keypoints[0], list) and len(keypoints[0]) == 3:
                        # Already in [[x,y,v], ...] format, flatten it
                        flat_keypoints = []
                        for kp in keypoints:
                            flat_keypoints.extend(kp)
                        obj["keypoints"] = flat_keypoints
                    else:
                        # Already flattened
                        obj["keypoints"] = keypoints
                    
                    # Ensure we have exactly 17 keypoints (51 values: 17 * 3)
                    if len(obj["keypoints"]) != 51:
                        # Silently fix keypoint format instead of verbose warning
                        # Some annotations have different keypoint formats
                        if len(obj["keypoints"]) < 51:
                            obj["keypoints"].extend([0.0] * (51 - len(obj["keypoints"])))
                        else:
                            obj["keypoints"] = obj["keypoints"][:51]
                
                # Add task-specific flags
                obj["has_bbox"] = "bbox" in obj and len(obj["bbox"]) == 4
                obj["has_segmentation"] = "segmentation" in obj
                obj["has_keypoints"] = "keypoints" in obj and len(obj["keypoints"]) == 51
                
                annotations.append(obj)
        
        if annotations:
            record["annotations"] = annotations
            dataset_dicts.append(record)
    
    logger.info(f"Created {len(dataset_dicts)} dataset records with multitask annotations")
    logger.info(f"Loading took {timer.seconds():.2f} seconds")
    
    return dataset_dicts


def validate_keypoints(keypoints, image_width, image_height):
    """
    Validate and clean keypoint annotations.
    
    Args:
        keypoints: list of keypoint coordinates [x1,y1,v1,x2,y2,v2,...]
        image_width: image width for bounds checking
        image_height: image height for bounds checking
        
    Returns:
        cleaned keypoints with valid coordinates
    """
    if len(keypoints) != 51:
        return None
    
    cleaned = []
    for i in range(0, 51, 3):
        x, y, v = keypoints[i:i+3]
        
        # Check if keypoint is valid
        if v > 0.1 and 0 <= x < image_width and 0 <= y < image_height:
            cleaned.extend([float(x), float(y), float(v)])
        else:
            # Mark as invisible/invalid
            cleaned.extend([0.0, 0.0, 0.0])
    
    return cleaned


def get_keypoint_statistics(dataset_dicts):
    """
    Compute statistics about keypoint annotations in the dataset.
    
    Args:
        dataset_dicts: list of dataset records
        
    Returns:
        dict with keypoint statistics
    """
    stats = {
        'total_instances': 0,
        'instances_with_keypoints': 0,
        'keypoint_visibility': [0] * 17,  # count for each of 17 keypoints
        'avg_visible_keypoints': 0
    }
    
    total_visible = 0
    
    for record in dataset_dicts:
        for ann in record["annotations"]:
            stats['total_instances'] += 1
            
            if ann.get("has_keypoints", False):
                stats['instances_with_keypoints'] += 1
                keypoints = ann["keypoints"]
                
                visible_count = 0
                for i in range(17):
                    v = keypoints[i * 3 + 2]  # visibility score
                    if v > 0.1:  # threshold for visible
                        stats['keypoint_visibility'][i] += 1
                        visible_count += 1
                
                total_visible += visible_count
    
    if stats['instances_with_keypoints'] > 0:
        stats['avg_visible_keypoints'] = total_visible / stats['instances_with_keypoints']
    
    return stats