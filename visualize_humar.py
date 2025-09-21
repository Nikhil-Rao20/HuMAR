"""
Visualization script for HuMAR multitask dataset.
This script loads data and visualizes detection boxes, segmentation masks, and keypoints.
"""

import sys
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import random

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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

# COCO skeleton connections (1-based indexing converted to 0-based)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],         # torso to arms
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],          # arms and face
    [1, 3], [2, 4], [3, 5], [4, 6]                    # face to shoulders
]

# Colors for different parts
KEYPOINT_COLORS = [
    [255, 0, 0],    # nose - red
    [255, 85, 0], [255, 170, 0],  # eyes - orange  
    [255, 255, 0], [170, 255, 0], # ears - yellow
    [85, 255, 0], [0, 255, 0],     # shoulders - green
    [0, 255, 85], [0, 255, 170],   # elbows - cyan
    [0, 255, 255], [0, 170, 255],  # wrists - light blue
    [0, 85, 255], [0, 0, 255],     # hips - blue
    [85, 0, 255], [170, 0, 255],   # knees - purple
    [255, 0, 255], [255, 0, 170]   # ankles - magenta
]

class HuMARVisualizer:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.gref_json = os.path.join(dataset_root, "HuMAR_Annots_With_Keypoints", "HuMAR_GREF_COCO_With_Keypoints.json")
        self.instances_json = os.path.join(dataset_root, "HuMAR_Annots_With_Keypoints", "HuMAR_instances_With_Keypoints.json")
        self.image_root = os.path.join(dataset_root, "GREFS_COCO_HuMAR_Images")
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load GREF and instances data."""
        print("Loading dataset...")
        
        # Load GREF data
        with open(self.gref_json, 'r') as f:
            self.gref_data = json.load(f)
        
        # Load instances (sample first for memory efficiency)
        print("Loading instances data (this may take a moment)...")
        with open(self.instances_json, 'r') as f:
            self.instances_data = json.load(f)
        
        # Create lookup table
        self.ann_id_to_instance = {ann['id']: ann for ann in self.instances_data}
        
        print(f"Loaded {len(self.gref_data)} referring expressions and {len(self.instances_data)} instances")
    
    def get_sample_data(self, num_samples=5, split="train"):
        """Get sample data for visualization."""
        # Filter by split
        filtered_gref = [item for item in self.gref_data if item.get('split', 'train') == split]
        
        # Group by image_id
        image_groups = {}
        for ref_item in filtered_gref:
            image_id = ref_item['image_id']
            if image_id not in image_groups:
                image_groups[image_id] = []
            image_groups[image_id].append(ref_item)
        
        # Sample random images
        sample_image_ids = random.sample(list(image_groups.keys()), min(num_samples, len(image_groups)))
        
        sample_data = []
        for image_id in sample_image_ids:
            refs = image_groups[image_id]
            
            # Get all instances for this image
            instances = []
            for ref in refs:
                for ann_id in ref['ann_id']:
                    if ann_id in self.ann_id_to_instance:
                        instance = self.ann_id_to_instance[ann_id].copy()
                        instance['ref_info'] = {
                            'ref_id': ref['ref_id'],
                            'sentences': ref['sentences']
                        }
                        instances.append(instance)
            
            if instances:
                sample_data.append({
                    'image_id': image_id,
                    'file_name': refs[0]['file_name'],
                    'instances': instances
                })
        
        return sample_data
    
    def visualize_sample(self, sample_data, save_dir="visualizations"):
        """Visualize sample data with all three tasks."""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, data in enumerate(sample_data):
            print(f"Visualizing sample {i+1}/{len(sample_data)}: {data['file_name']}")
            
            # Load image
            image_path = os.path.join(self.image_root, data['file_name'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'HuMAR Multitask Visualization: {data["file_name"]}', fontsize=16)
            
            # 1. Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # 2. Detection boxes
            axes[0, 1].imshow(image)
            self.draw_detection_boxes(axes[0, 1], data['instances'])
            axes[0, 1].set_title("Detection Boxes")
            axes[0, 1].axis('off')
            
            # 3. Segmentation masks
            axes[1, 0].imshow(image)
            self.draw_segmentation_masks(axes[1, 0], data['instances'], image.shape)
            axes[1, 0].set_title("Segmentation Masks")
            axes[1, 0].axis('off')
            
            # 4. Keypoints
            axes[1, 1].imshow(image)
            self.draw_keypoints(axes[1, 1], data['instances'])
            axes[1, 1].set_title("Keypoints")
            axes[1, 1].axis('off')
            
            # Add text information
            self.add_text_info(fig, data['instances'])
            
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(save_dir, f"sample_{i+1}_{data['image_id']}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization: {save_path}")
    
    def draw_detection_boxes(self, ax, instances):
        """Draw bounding boxes on the axes."""
        for instance in instances:
            bbox = instance.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add referring expression
                ref_info = instance.get('ref_info', {})
                if ref_info and ref_info.get('sentences'):
                    text = ref_info['sentences'][0]['raw'][:50] + "..."
                    ax.text(x, y-10, text, fontsize=8, color='red', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    def draw_segmentation_masks(self, ax, instances, image_shape):
        """Draw segmentation masks on the axes."""
        for idx, instance in enumerate(instances):
            segmentation = instance.get('segmentation', [])
            if segmentation:
                # Handle polygon format
                if isinstance(segmentation, list) and segmentation:
                    for poly in segmentation:
                        if len(poly) >= 6:  # At least 3 points (x,y pairs)
                            poly_array = np.array(poly).reshape(-1, 2)
                            # Create a random color for each instance
                            color = plt.cm.Set3(idx / max(len(instances), 1))
                            polygon = patches.Polygon(poly_array, closed=True, 
                                                    alpha=0.5, facecolor=color, edgecolor='black')
                            ax.add_patch(polygon)
    
    def draw_keypoints(self, ax, instances):
        """Draw keypoints and skeleton on the axes."""
        for instance in instances:
            keypoints = instance.get('keypoints', [])
            if len(keypoints) == 17:  # 17 keypoints in [[x,y,v], ...] format
                self.draw_keypoints_on_image(ax, keypoints)
    
    def draw_keypoints_on_image(self, ax, keypoints):
        """Draw individual keypoints and skeleton."""
        # Convert keypoints to proper format if needed
        if len(keypoints) == 17 and isinstance(keypoints[0], list):
            # Already in [[x,y,v], ...] format
            kpts = keypoints
        elif len(keypoints) == 51:
            # Flat format [x1,y1,v1,x2,y2,v2,...]
            kpts = []
            for i in range(0, 51, 3):
                kpts.append([keypoints[i], keypoints[i+1], keypoints[i+2]])
        else:
            return  # Invalid format
        
        # Draw keypoints
        visible_keypoints = []
        for i, (x, y, v) in enumerate(kpts):
            if v > 0.1:  # Visibility threshold
                color = np.array(KEYPOINT_COLORS[i]) / 255.0
                ax.scatter(x, y, c=[color], s=50, alpha=0.8)
                ax.text(x+3, y+3, COCO_KEYPOINT_NAMES[i], fontsize=6, color='white',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
                visible_keypoints.append((i, x, y))
        
        # Draw skeleton connections
        for connection in COCO_SKELETON:
            pt1_idx, pt2_idx = connection[0], connection[1]
            
            # Find if both keypoints are visible
            pt1, pt2 = None, None
            for idx, x, y in visible_keypoints:
                if idx == pt1_idx:
                    pt1 = (x, y)
                elif idx == pt2_idx:
                    pt2 = (x, y)
            
            if pt1 and pt2:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2, alpha=0.7)
    
    def add_text_info(self, fig, instances):
        """Add text information about the instances."""
        info_text = f"Instances: {len(instances)}\n"
        
        detection_count = sum(1 for inst in instances if inst.get('bbox'))
        segmentation_count = sum(1 for inst in instances if inst.get('segmentation'))
        keypoint_count = sum(1 for inst in instances if inst.get('keypoints'))
        
        info_text += f"Detection: {detection_count}\n"
        info_text += f"Segmentation: {segmentation_count}\n"
        info_text += f"Keypoints: {keypoint_count}\n"
        
        # Add referring expressions
        if instances:
            ref_info = instances[0].get('ref_info', {})
            if ref_info and ref_info.get('sentences'):
                info_text += f"\nReferring expressions:\n"
                for sentence in ref_info['sentences'][:2]:  # Show first 2
                    info_text += f"- {sentence['raw']}\n"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


def main():
    """Main function to run visualization."""
    dataset_root = r"c:\Users\nikhi\Desktop\ReLA\datasets"
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    # Initialize visualizer
    visualizer = HuMARVisualizer(dataset_root)
    
    # Get sample data
    print("Getting sample data...")
    sample_data = visualizer.get_sample_data(num_samples=3, split="train")
    
    if not sample_data:
        print("Error: No sample data found")
        return
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer.visualize_sample(sample_data)
    
    print("✓ Visualization completed!")
    print("Check the 'visualizations' directory for output images.")


if __name__ == "__main__":
    main()