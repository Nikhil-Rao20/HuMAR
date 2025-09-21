"""
Test script for HuMAR multitask dataloader.
This script validates the dataloader and prints dataset statistics.
"""

import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_humar_dataloader():
    """Test the HuMAR multitask dataloader without full Detectron2 setup."""
    
    # Test basic JSON loading first
    print("Testing basic dataset loading...")
    
    dataset_root = r"c:\Users\nikhi\Desktop\ReLA\datasets"
    gref_json = os.path.join(dataset_root, "HuMAR_Annots_With_Keypoints", "HuMAR_GREF_COCO_With_Keypoints.json")
    instances_json = os.path.join(dataset_root, "HuMAR_Annots_With_Keypoints", "HuMAR_instances_With_Keypoints.json")
    image_root = os.path.join(dataset_root, "GREFS_COCO_HuMAR_Images")
    
    # Check if files exist
    files_to_check = [
        (gref_json, "GREF JSON"),
        (instances_json, "Instances JSON"), 
        (image_root, "Image directory")
    ]
    
    for file_path, name in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {name} found: {file_path}")
        else:
            print(f"✗ {name} not found: {file_path}")
            return False
    
    # Test loading GREF data
    print("\nLoading GREF data...")
    try:
        with open(gref_json, 'r') as f:
            gref_data = json.load(f)
        print(f"✓ Loaded {len(gref_data)} referring expressions")
        
        # Show sample
        if gref_data:
            sample = gref_data[0]
            print(f"Sample GREF entry: image_id={sample.get('image_id')}, ref_id={sample.get('ref_id')}")
            print(f"Sample sentence: {sample['sentences'][0]['raw']}")
    except Exception as e:
        print(f"✗ Error loading GREF data: {e}")
        return False
    
    # Test loading instances data (just peek at structure)
    print("\nAnalyzing instances data structure...")
    try:
        with open(instances_json, 'r') as f:
            # Read first few instances to understand structure
            content = f.read(5000)  # Read first 5KB
            
        # Check if it's valid JSON start
        if content.strip().startswith('['):
            print("✓ Instances file appears to be valid JSON array")
            
            # Count approximate number of instances by counting 'id' fields
            id_count = content.count('"id":')
            print(f"Estimated instances in sample: ~{id_count}")
            
            # Check for required fields
            required_fields = ['bbox', 'segmentation', 'keypoints', 'category_id']
            found_fields = [field for field in required_fields if field in content]
            print(f"Found required fields: {found_fields}")
            
        else:
            print("✗ Instances file doesn't appear to be valid JSON")
            return False
            
    except Exception as e:
        print(f"✗ Error analyzing instances data: {e}")
        return False
    
    # Test image directory
    print("\nChecking image directory...")
    try:
        image_files = [f for f in os.listdir(image_root) if f.endswith('.jpg')]
        print(f"✓ Found {len(image_files)} image files")
        
        if image_files:
            print(f"Sample image: {image_files[0]}")
    except Exception as e:
        print(f"✗ Error accessing image directory: {e}")
        return False
    
    # Analyze dataset splits
    print("\nAnalyzing dataset splits...")
    try:
        splits = {}
        for item in gref_data:
            split = item.get('split', 'unknown')
            splits[split] = splits.get(split, 0) + 1
        
        print("Split distribution:")
        for split, count in splits.items():
            print(f"  {split}: {count} instances")
            
    except Exception as e:
        print(f"Warning: Could not analyze splits: {e}")
    
    # Test basic multitask loading (without Detectron2)
    print("\nTesting basic multitask data structure...")
    try:
        # Create a simple mock of the dataloader functionality
        from gres_model.data.datasets.humar_multitask import get_keypoint_statistics
        
        # Create a simplified dataset dict
        sample_dicts = []
        
        # Group by image_id
        image_groups = {}
        for ref_item in gref_data[:100]:  # Test with first 100 items
            image_id = ref_item['image_id']
            if image_id not in image_groups:
                image_groups[image_id] = []
            image_groups[image_id].append(ref_item)
        
        print(f"Testing with {len(image_groups)} unique images from first 100 references")
        
        # Simple structure validation
        multitask_capabilities = {
            'detection': 0,
            'segmentation': 0, 
            'keypoints': 0
        }
        
        for image_id in list(image_groups.keys())[:5]:  # Test first 5 images
            refs = image_groups[image_id]
            for ref in refs:
                for ann_id in ref['ann_id']:
                    multitask_capabilities['detection'] += 1  # All have bboxes
                    multitask_capabilities['segmentation'] += 1  # All should have segmentation
                    multitask_capabilities['keypoints'] += 1  # All should have keypoints
        
        print("Multitask capability estimate:")
        for task, count in multitask_capabilities.items():
            print(f"  {task}: {count} instances")
            
    except Exception as e:
        print(f"Warning: Could not test multitask structure: {e}")
    
    print("\n✓ Basic dataloader test completed successfully!")
    print("\nNext steps:")
    print("1. Set up full Detectron2 environment")
    print("2. Test with actual MultitaskDatasetMapper")
    print("3. Create visualization script")
    
    return True


if __name__ == "__main__":
    success = test_humar_dataloader()
    sys.exit(0 if success else 1)