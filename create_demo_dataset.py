#!/usr/bin/env python3
"""
Create a small demo RefCOCO dataset for testing the training pipeline.
This creates synthetic data that matches the expected RefCOCO format.
"""

import os
import json
import pickle
from collections import defaultdict

def create_demo_refcoco_dataset():
    """Create a minimal demo RefCOCO dataset for testing"""
    
    # Create the refer dataset directory structure
    dataset_dir = "datasets/refcoco"
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("🔧 Creating demo RefCOCO dataset...")
    
    # Create synthetic refs data that matches REFER class expectations
    refs_data = {
        'dataset': 'refcoco',
        'refs': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object', 'supercategory': 'thing'}]
    }
    
    # Create synthetic data for 4 splits with minimal samples each
    splits = {
        'train': list(range(1, 21)),      # 20 samples
        'val': list(range(21, 31)),       # 10 samples  
        'testA': list(range(31, 41)),     # 10 samples
        'testB': list(range(41, 51))      # 10 samples
    }
    
    ref_id = 1
    ann_id = 1
    
    for split_name, image_ids in splits.items():
        for img_id in image_ids:
            # Create image entry
            image_entry = {
                'id': img_id,
                'width': 640,
                'height': 480,
                'file_name': f'demo_{img_id:06d}.jpg',
                'split': split_name
            }
            refs_data['images'].append(image_entry)
            
            # Create annotation entry
            ann_entry = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'segmentation': [[[100, 100, 200, 100, 200, 200, 100, 200]]],  # Simple box
                'area': 10000,
                'bbox': [100, 100, 100, 100],  # [x, y, width, height]
                'iscrowd': 0
            }
            refs_data['annotations'].append(ann_entry)
            
            # Create referring expression entries
            for sent_id in range(3):  # 3 sentences per image
                ref_entry = {
                    'ref_id': ref_id,
                    'ann_id': ann_id,
                    'image_id': img_id,
                    'split': split_name,
                    'sentences': [
                        {'sent': f'demo object {ref_id} sentence {sent_id}', 
                         'sent_id': ref_id * 10 + sent_id}
                    ]
                }
                refs_data['refs'].append(ref_entry)
                ref_id += 1
            
            ann_id += 1
    
    # Save the refs pickle file that REFER expects
    refs_file = os.path.join(dataset_dir, 'refs(unc).p')
    with open(refs_file, 'wb') as f:
        pickle.dump(refs_data, f)
    
    print(f"✅ Created refs file: {refs_file}")
    print(f"   - Total images: {len(refs_data['images'])}")
    print(f"   - Total annotations: {len(refs_data['annotations'])}")
    print(f"   - Total refs: {len(refs_data['refs'])}")
    
    # Also save instances.json that REFER class expects
    instances_data = {
        'images': refs_data['images'],
        'annotations': refs_data['annotations'],
        'categories': refs_data['categories']
    }
    
    instances_file = os.path.join(dataset_dir, 'instances.json')
    with open(instances_file, 'w') as f:
        json.dump(instances_data, f)
    
    print(f"✅ Created instances file: {instances_file}")
    
    # Create split mappings that REFER expects
    split_mapping = defaultdict(list)
    for ref in refs_data['refs']:
        split_mapping[ref['split']].append(ref['ref_id'])
    
    # Save split files
    for split_name, ref_ids in split_mapping.items():
        split_file = os.path.join(dataset_dir, f'{split_name}.txt')
        with open(split_file, 'w') as f:
            for ref_id in ref_ids:
                f.write(f'{ref_id}\n')
        print(f"✅ Created split file: {split_file} ({len(ref_ids)} refs)")
    
    # Create dummy image directory
    image_dir = "datasets/images/train2014"
    os.makedirs(image_dir, exist_ok=True)
    
    # Create a simple 1x1 dummy image for testing (in binary)
    dummy_jpg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        # ... minimal JPEG header ...
        0xFF, 0xD9  # JPEG end marker
    ])
    
    # Create dummy images for all our demo data
    for split_name, image_ids in splits.items():
        for img_id in image_ids:
            img_path = os.path.join(image_dir, f'demo_{img_id:06d}.jpg')
            with open(img_path, 'wb') as f:
                f.write(dummy_jpg_bytes)
    
    print(f"✅ Created {sum(len(ids) for ids in splits.values())} dummy images in {image_dir}")
    
    return dataset_dir

if __name__ == "__main__":
    create_demo_refcoco_dataset()
    print("\n🎉 Demo RefCOCO dataset created successfully!")
    print("Now you can run training with the demo dataset.")