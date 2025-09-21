# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from .humar_multitask import load_humar_multitask_json, COCO_KEYPOINT_NAMES, COCO_KEYPOINT_SKELETON


def register_humar_multitask(root):
    """
    Register HuMAR multitask dataset for detection, segmentation, and pose estimation.
    
    Args:
        root (str): Root directory containing the dataset
    """
    image_root = os.path.join(root, "GREFS_COCO_HuMAR_Images")
    gref_json = os.path.join(root, "HuMAR_Annots_With_Keypoints", "HuMAR_GREF_COCO_With_Keypoints.json")
    instances_json = os.path.join(root, "HuMAR_Annots_With_Keypoints", "HuMAR_instances_With_Keypoints.json")
    
    # Register train dataset
    dataset_name = "humar_multitask_train"
    
    # Check if dataset is already registered
    try:
        DatasetCatalog.get(dataset_name)
        print(f"Dataset {dataset_name} already registered, skipping...")
    except KeyError:
        DatasetCatalog.register(
            dataset_name,
            lambda: load_humar_multitask_json(
                gref_json_file=gref_json,
                instances_json_file=instances_json,
                image_root=image_root,
                split="train"
            )
        )
        
        # Set metadata for the dataset
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["person"],  # Only person class for human-centric dataset
            keypoint_names=COCO_KEYPOINT_NAMES,
            keypoint_flip_map=_get_coco_keypoint_flip_map(),
            keypoint_connection_rules=COCO_KEYPOINT_SKELETON,
            evaluator_type="multitask",  # Custom evaluator for multitask
            dataset_name="humar_multitask",
            split="train",
            root=root,
            image_root=image_root,
            task_types=["detection", "segmentation", "keypoints"],
            num_keypoints=17,
        )
    
    # Register validation dataset (if available)
    val_dataset_name = "humar_multitask_val"
    
    try:
        DatasetCatalog.get(val_dataset_name)
        print(f"Dataset {val_dataset_name} already registered, skipping...")
    except KeyError:
        DatasetCatalog.register(
            val_dataset_name,
            lambda: load_humar_multitask_json(
                gref_json_file=gref_json,
                instances_json_file=instances_json,
                image_root=image_root,
                split="val"
            )
        )
        
        MetadataCatalog.get(val_dataset_name).set(
            thing_classes=["person"],
            keypoint_names=COCO_KEYPOINT_NAMES,
            keypoint_flip_map=_get_coco_keypoint_flip_map(),
            keypoint_connection_rules=COCO_KEYPOINT_SKELETON,
            evaluator_type="multitask",
            dataset_name="humar_multitask",
            split="val",
            root=root,
            image_root=image_root,
        task_types=["detection", "segmentation", "keypoints"],
        num_keypoints=17,
    )


def _get_coco_keypoint_flip_map():
    """
    Get the keypoint flip map for data augmentation.
    Maps left keypoints to right keypoints and vice versa.
    """
    return [
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"), 
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]


def register_humar_datasets():
    """
    Register all HuMAR datasets with default path.
    """
    # Try to get dataset root from environment variable
    root = os.getenv("DETECTRON2_DATASETS", "datasets")
    
    # Check if HuMAR dataset exists
    humar_root = os.path.join(root)
    if os.path.exists(humar_root):
        register_humar_multitask(humar_root)
    else:
        print(f"Warning: HuMAR dataset not found at {humar_root}")


# Auto-register when module is imported
if __name__ != "__main__":
    register_humar_datasets()