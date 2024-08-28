import json
import random
import os

# Paths to your input and output JSON files
instances_train_path = '/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train.json'  # Replace with your path
output_dir = '/home/djones/puncta_det/data_puncta/active_learning'  # Replace with your path

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the instances_train.json file
with open(instances_train_path, 'r') as f:
    coco_data = json.load(f)

# Number of labeled and unlabeled splits
n_splits = 3  # Change this if you have a different number of splits
labeled_per_split = 600  # Number of labeled images per split, adjusted for a smaller dataset

# Get all image IDs
image_ids = [image['id'] for image in coco_data['images']]
random.shuffle(image_ids)  # Shuffle image IDs for random split

# Split into labeled and unlabeled datasets
for i in range(1, n_splits + 1):
    # Calculate indices for this split
    labeled_start_idx = (i - 1) * labeled_per_split
    labeled_end_idx = i * labeled_per_split

    # Split the data into labeled and unlabeled
    labeled_image_ids = set(image_ids[labeled_start_idx:labeled_end_idx])
    unlabeled_image_ids = set(image_ids) - labeled_image_ids

    # Create labeled dataset
    labeled_data = {
        "images": [img for img in coco_data['images'] if img['id'] in labeled_image_ids],
        "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in labeled_image_ids],
        "categories": coco_data['categories']
    }

    # Create unlabeled dataset
    unlabeled_data = {
        "images": [img for img in coco_data['images'] if img['id'] in unlabeled_image_ids],
        "annotations": [],  # No annotations for unlabeled data
        "categories": coco_data['categories']
    }

    # Save the labeled JSON file
    labeled_json_path = os.path.join(output_dir, f'coco_{labeled_per_split}_labeled_{i}.json')
    with open(labeled_json_path, 'w') as f:
        json.dump(labeled_data, f)

    # Save the unlabeled JSON file
    unlabeled_json_path = os.path.join(output_dir, f'coco_{labeled_per_split}_unlabeled_{i}.json')
    with open(unlabeled_json_path, 'w') as f:
        json.dump(unlabeled_data, f)

    print(f"Labeled JSON saved to {labeled_json_path}")
    print(f"Unlabeled JSON saved to {unlabeled_json_path}")
