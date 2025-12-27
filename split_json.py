import json
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Load the original JSON file
print("Loading JSON file...")
with open('RefHuman_Training/RefHuman_train.json', 'r') as f:
    data = json.load(f)

# Extract data
info = data['info']
licenses = data['licenses']
images = data['images']
annotations = data['annotations']
categories = data['categories']

print(f"Total images: {len(images)}")
print(f"Total annotations: {len(annotations)}")

# Shuffle images for random split
shuffled_images = images.copy()
random.shuffle(shuffled_images)

# Split images into train (80%) and val (20%)
split_idx = int(0.8 * len(shuffled_images))
train_images = shuffled_images[:split_idx]
val_images = shuffled_images[split_idx:]

print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")

# Create a set of train and val image IDs for quick lookup
train_image_ids = {img['id'] for img in train_images}
val_image_ids = {img['id'] for img in val_images}

# Distribute annotations based on image split
train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

print(f"Train annotations: {len(train_annotations)}")
print(f"Val annotations: {len(val_annotations)}")

# Create train split
train_data = {
    'info': info,
    'licenses': licenses,
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}

# Create val split
val_data = {
    'info': info,
    'licenses': licenses,
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

# Save train split
print("Saving train split...")
with open('RefHuman_Training/RefHuman_train.json', 'w') as f:
    json.dump(train_data, f)
print("✓ Saved RefHuman_train.json")

# Save val split
print("Saving val split...")
with open('RefHuman_Training/RefHuman_val.json', 'w') as f:
    json.dump(val_data, f)
print("✓ Saved RefHuman_val.json")

print("\nSplit complete!")
