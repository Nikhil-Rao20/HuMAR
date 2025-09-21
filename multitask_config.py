"""
Configuration for MultitaskGRES model.
Extends the base ReLA configuration with multitask-specific settings.
"""

# Base configuration template for multitask GRES
MULTITASK_CONFIG = {
    # Model configuration
    "MODEL": {
        # Enable multitask heads
        "ENABLE_DETECTION": True,
        "ENABLE_KEYPOINTS": True,
        
        # Head dimensions
        "DETECTION_HEAD_DIM": 256,
        "KEYPOINT_HEAD_DIM": 256,
        
        # Base MASK_FORMER settings
        "MASK_FORMER": {
            "NUM_OBJECT_QUERIES": 100,
            "HIDDEN_DIM": 256,
            "CLASS_WEIGHT": 1.0,
            "DICE_WEIGHT": 1.0,
            "MASK_WEIGHT": 1.0,
            "TEST": {
                "OBJECT_MASK_THRESHOLD": 0.8,
                "OVERLAP_THRESHOLD": 0.8,
                "SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE": True,
                "SEMANTIC_ON": False,
                "INSTANCE_ON": False,
                "PANOPTIC_ON": False,
            },
            "SIZE_DIVISIBILITY": 32,
        },
        
        # Pixel mean and std (ImageNet)
        "PIXEL_MEAN": [123.675, 116.28, 103.53],
        "PIXEL_STD": [58.395, 57.12, 57.375],
    },
    
    # Training configuration
    "SOLVER": {
        "BASE_LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MAX_ITER": 90000,
        "WARMUP_ITERS": 1000,
        "WARMUP_FACTOR": 0.001,
        "CLIP_GRADIENTS": {
            "ENABLED": True,
            "CLIP_TYPE": "norm",
            "CLIP_VALUE": 1.0,
        },
        "LR_SCHEDULER_NAME": "WarmupCosineLR",
    },
    
    # Dataset configuration
    "DATASETS": {
        "TRAIN": ["humar_multitask_train"],
        "TEST": ["humar_multitask_val"],
    },
    
    # DataLoader configuration
    "DATALOADER": {
        "NUM_WORKERS": 4,
        "BATCH_SIZE_TRAIN": 8,
        "BATCH_SIZE_TEST": 4,
    },
    
    # Input configuration
    "INPUT": {
        "MIN_SIZE_TRAIN": (320, 384, 448, 512, 576, 640, 704, 768),
        "MIN_SIZE_TEST": 512,
        "MAX_SIZE_TRAIN": 1024,
        "MAX_SIZE_TEST": 1024,
        "FORMAT": "RGB",
    },
    
    # Test configuration
    "TEST": {
        "DETECTIONS_PER_IMAGE": 100,
    },
    
    # Referring expression configuration
    "REFERRING": {
        "BERT_TYPE": "bert-base-uncased",
        "MAX_QUERY_LEN": 77,
    },
    
    # Multitask loss weights
    "LOSS_WEIGHTS": {
        "SEGMENTATION": 1.0,
        "DETECTION": 1.0,
        "KEYPOINTS": 1.0,
        
        # Detection sub-weights
        "DET_CLASSIFICATION": 1.0,
        "DET_BBOX_REGRESSION": 5.0,
        
        # Keypoint sub-weights
        "KPT_COORDINATE": 1.0,
        "KPT_VISIBILITY": 1.0,
    },
    
    # Output directory
    "OUTPUT_DIR": "./output_multitask",
    
    # Checkpoint and logging
    "CHECKPOINT_PERIOD": 5000,
    "LOG_PERIOD": 100,
    "EVAL_PERIOD": 5000,
    
    # Visualization
    "VIS_PERIOD": 1000,
    "NUM_VIS_SAMPLES": 4,
}


def get_multitask_config():
    """Get multitask configuration."""
    return MULTITASK_CONFIG.copy()


def update_config_for_dataset(config, dataset_path, num_classes=1):
    """Update configuration for specific dataset."""
    config = config.copy()
    
    # Update dataset paths
    config["DATASETS"]["TRAIN"] = [f"humar_multitask_train"]
    config["DATASETS"]["TEST"] = [f"humar_multitask_val"]
    
    # Update model for number of classes
    config["MODEL"]["NUM_CLASSES"] = num_classes
    
    return config


def get_training_config():
    """Get configuration optimized for training."""
    config = get_multitask_config()
    
    # Training-specific settings
    config["MODEL"]["MASK_FORMER"]["TEST"]["SEMANTIC_ON"] = False
    config["MODEL"]["MASK_FORMER"]["TEST"]["INSTANCE_ON"] = True
    config["MODEL"]["MASK_FORMER"]["TEST"]["PANOPTIC_ON"] = False
    
    return config


def get_inference_config():
    """Get configuration optimized for inference."""
    config = get_multitask_config()
    
    # Inference-specific settings
    config["MODEL"]["MASK_FORMER"]["TEST"]["SEMANTIC_ON"] = True
    config["MODEL"]["MASK_FORMER"]["TEST"]["INSTANCE_ON"] = True
    config["MODEL"]["MASK_FORMER"]["TEST"]["PANOPTIC_ON"] = False
    
    # Lower thresholds for better recall
    config["MODEL"]["MASK_FORMER"]["TEST"]["OBJECT_MASK_THRESHOLD"] = 0.5
    config["MODEL"]["MASK_FORMER"]["TEST"]["OVERLAP_THRESHOLD"] = 0.5
    
    return config


# COCO keypoint configuration
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_KEYPOINT_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# Model architecture settings
BACKBONE_CONFIGS = {
    "resnet50": {
        "TYPE": "ResNet",
        "DEPTH": 50,
        "OUT_FEATURES": ["res2", "res3", "res4", "res5"],
    },
    "swin_tiny": {
        "TYPE": "SwinTransformer", 
        "EMBED_DIM": 96,
        "DEPTHS": [2, 2, 6, 2],
        "NUM_HEADS": [3, 6, 12, 24],
        "WINDOW_SIZE": 7,
        "MLP_RATIO": 4.0,
        "QKV_BIAS": True,
        "QK_SCALE": None,
        "DROP_RATE": 0.0,
        "ATTN_DROP_RATE": 0.0,
        "DROP_PATH_RATE": 0.3,
        "APE": False,
        "PATCH_NORM": True,
        "OUT_FEATURES": ["res2", "res3", "res4", "res5"],
    },
    "swin_base": {
        "TYPE": "SwinTransformer",
        "EMBED_DIM": 128,
        "DEPTHS": [2, 2, 18, 2],
        "NUM_HEADS": [4, 8, 16, 32],
        "WINDOW_SIZE": 7,
        "MLP_RATIO": 4.0,
        "QKV_BIAS": True,
        "QK_SCALE": None,
        "DROP_RATE": 0.0,
        "ATTN_DROP_RATE": 0.0,
        "DROP_PATH_RATE": 0.5,
        "APE": False,
        "PATCH_NORM": True,
        "OUT_FEATURES": ["res2", "res3", "res4", "res5"],
    }
}


def get_config_with_backbone(backbone_name="resnet50"):
    """Get configuration with specific backbone."""
    config = get_multitask_config()
    
    if backbone_name in BACKBONE_CONFIGS:
        config["MODEL"]["BACKBONE"] = BACKBONE_CONFIGS[backbone_name]
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    return config


if __name__ == "__main__":
    # Example usage
    import json
    
    print("MultitaskGRES Configuration")
    print("=" * 40)
    
    # Get base config
    config = get_multitask_config()
    
    print("Base configuration:")
    print(json.dumps(config, indent=2))
    
    print("\nAvailable backbones:")
    for backbone in BACKBONE_CONFIGS.keys():
        print(f"- {backbone}")
    
    print(f"\nKeypoint names ({len(COCO_KEYPOINT_NAMES)}):")
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        print(f"{i+1:2d}. {name}")
    
    print(f"\nKeypoint skeleton ({len(COCO_KEYPOINT_SKELETON)} connections):")
    for i, (start, end) in enumerate(COCO_KEYPOINT_SKELETON):
        start_name = COCO_KEYPOINT_NAMES[start-1]
        end_name = COCO_KEYPOINT_NAMES[end-1]
        print(f"{i+1:2d}. {start_name} -> {end_name}")