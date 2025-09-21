#!/usr/bin/env python3
"""
HuMAR-GREF dataset loader for ReLA training.
Loads the actual HuMAR_GREF_COCO_With_Keypoints.json and HuMAR_instances_With_Keypoints.json files.
"""

import json
import os
import sys
from collections import defaultdict
from PIL import Image

# Add local detectron2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def load_humar_gref_annotations(annotations_root, images_root, split):
    """
    Load HuMAR-GREF dataset annotations for the specified split.
    
    Args:
        annotations_root: Path to HuMAR_Annots_With_Keypoints directory
        images_root: Path to GREFS_COCO_HuMAR_Images directory  
        split: Data split ('train', 'val', 'testA', 'testB')
    
    Returns:
        List of annotation dictionaries in Detectron2 format
    """
    
    # Load referring expressions
    gref_file = os.path.join(annotations_root, "HuMAR_GREF_COCO_With_Keypoints.json")
    with open(gref_file, 'r') as f:
        gref_data = json.load(f)
    
    # Load instance annotations  
    instances_file = os.path.join(annotations_root, "HuMAR_instances_With_Keypoints.json")
    with open(instances_file, 'r') as f:
        instances_data = json.load(f)
    
    # Create mapping from annotation ID to instance data
    ann_id_to_instance = {inst['id']: inst for inst in instances_data}
    
    # Filter referring expressions by split
    split_refs = [ref for ref in gref_data if ref['split'] == split]
    
    # Group by image_id
    image_refs = defaultdict(list)
    for ref in split_refs:
        image_refs[ref['image_id']].append(ref)
    
    dataset_dicts = []
    
    for image_id, refs in image_refs.items():
        # Get image info from first ref (all refs for same image have same image info)
        sample_ref = refs[0]
        file_name = sample_ref['file_name']
        
        # Construct full image path
        image_path = os.path.join(images_root, file_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get actual image dimensions
        try:
            with Image.open(image_path) as img:
                actual_width, actual_height = img.size
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            # Use default dimensions as fallback
            actual_width, actual_height = 640, 480
            
        # Create image record with actual dimensions
        record = {
            "file_name": image_path,
            "image_id": image_id,
            "height": actual_height,  # Use actual image height
            "width": actual_width,    # Use actual image width
            "annotations": []
        }
        
        # Process each referring expression for this image
        for ref in refs:
            # Get all annotation IDs for this referring expression
            ann_ids = ref['ann_id']
            
            for ann_id in ann_ids:
                if ann_id not in ann_id_to_instance:
                    print(f"Warning: Annotation {ann_id} not found in instances")
                    continue
                    
                instance = ann_id_to_instance[ann_id]
                
                # Convert instance to Detectron2 format
                obj = {
                    "bbox": instance['bbox'],  # [x, y, width, height]
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": instance['segmentation'],
                    "category_id": instance['category_id'] - 1,  # Convert to 0-based indexing
                    "area": instance['area'],
                    "iscrowd": instance['iscrowd'],
                    
                    # Add referring expression data
                    "ref_id": ref['ref_id'],
                    "sentences": ref['sentences'],
                    "sent_ids": ref['sent_ids'],
                    
                    # Add keypoints if available
                    "keypoints": instance.get('keypoints', []),
                    "num_keypoints": len(instance.get('keypoints', [])) // 3 if instance.get('keypoints') else 0,
                }
                
                record["annotations"].append(obj)
        
        # Only add images that have annotations
        if record["annotations"]:
            dataset_dicts.append(record)
    
    print(f"Loaded {len(dataset_dicts)} images with annotations for split '{split}'")
    return dataset_dicts


def register_humar_gref_datasets():
    """Register HuMAR-GREF datasets for all available splits."""
    
    # Get dataset paths
    datasets_root = os.getenv("DETECTRON2_DATASETS", "datasets")
    annotations_root = os.path.join(datasets_root, "HuMAR_Annots_With_Keypoints")
    images_root = os.path.join(datasets_root, "GREFS_COCO_HuMAR_Images")
    
    # Available splits based on the analysis
    splits = ["train", "val", "testA", "testB"]
    
    for split in splits:
        dataset_name = f"humar_gref_{split}"
        
        # Register dataset
        DatasetCatalog.register(
            dataset_name,
            lambda split=split: load_humar_gref_annotations(annotations_root, images_root, split)
        )
        
        # Set metadata
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["person"],  # HuMAR-GREF focuses on people
            evaluator_type="refer",
            dataset_name="humar_gref",
            split=split,
            annotations_root=annotations_root,
            images_root=images_root,
            keypoint_names=[
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            keypoint_flip_map=[
                ("left_eye", "right_eye"),
                ("left_ear", "right_ear"), 
                ("left_shoulder", "right_shoulder"),
                ("left_elbow", "right_elbow"),
                ("left_wrist", "right_wrist"),
                ("left_hip", "right_hip"),
                ("left_knee", "right_knee"),
                ("left_ankle", "right_ankle"),
            ],
        )
        
        print(f"✅ Registered dataset: {dataset_name}")


if __name__ == "__main__":
    # Register the datasets
    register_humar_gref_datasets()
    
    # Test loading a sample
    test_data = load_humar_gref_annotations(
        "datasets/HuMAR_Annots_With_Keypoints",
        "datasets/GREFS_COCO_HuMAR_Images", 
        "testA"
    )
    print(f"\n🎯 Test load successful: {len(test_data)} samples for testA split")
    if test_data:
        sample = test_data[0]
        print(f"Sample image: {sample['file_name']}")
        print(f"Sample has {len(sample['annotations'])} annotations")
        if sample['annotations']:
            ann = sample['annotations'][0]
            print(f"Sample annotation: ref_id={ann.get('ref_id')}, bbox={ann['bbox']}")
            print(f"Sample sentence: {ann['sentences'][0]['sent'] if ann.get('sentences') else 'N/A'}")