import os
import json

def validate_coco_format(coco_json_path, image_dir):
    """
    Validates the COCO annotations JSON file and ensures that:
    1. Every image listed in the JSON file has corresponding annotations.
    2. The JSON file follows the COCO format.
    3. Every annotation refers to an existing image.

    Args:
        coco_json_path (str): Path to the COCO JSON file.
        image_dir (str): Directory where images are stored.

    Returns:
        bool: True if validation is successful, False otherwise.
    """
    # Load COCO JSON
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading COCO JSON: {e}")
        return False

    # Check mandatory fields in COCO format
    mandatory_fields = ['images', 'annotations', 'categories']
    for field in mandatory_fields:
        if field not in coco_data:
            print(f"COCO JSON is missing mandatory field: {field}")
            return False

    # Check that images have valid file names and annotations
    image_ids = set()
    image_files = set()
    
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        
        # Check if image file exists
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Image file {file_name} does not exist in {image_dir}")
            return False
        
        image_ids.add(image_id)
        image_files.add(file_name)
    
    # Check annotations
    for annotation in coco_data['annotations']:
        if 'image_id' not in annotation or annotation['image_id'] not in image_ids:
            print(f"Annotation refers to non-existent image_id: {annotation.get('image_id')}")
            return False
        
        if 'bbox' not in annotation or len(annotation['bbox']) != 4:
            print(f"Annotation missing or incorrect bbox format: {annotation}")
            return False
        
        if 'category_id' not in annotation:
            print(f"Annotation missing category_id: {annotation}")
            return False
    
    # Check categories
    if not coco_data['categories']:
        print("No categories found in COCO JSON.")
        return False
    
    # Check that each image has at least one annotation
    annotations_by_image = {image_id: 0 for image_id in image_ids}
    for annotation in coco_data['annotations']:
        annotations_by_image[annotation['image_id']] += 1
    
    for image_id, count in annotations_by_image.items():
        if count == 0:
            print(f"Image ID {image_id} has no annotations.")
            return False
    
    print("COCO validation successful!")
    return True

# Define paths
train_coco_json = '/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train2017.json'
val_coco_json = '/home/djones/puncta_det/data_puncta/puncta/annotations/instances_val2017.json'
train_image_dir = '/home/djones/puncta_det/data_puncta/puncta/train_image_dir'
val_image_dir = '/home/djones/puncta_det/data_puncta/puncta/val_image_dir'

# Validate training data
print("Validating training data:")
train_valid = validate_coco_format(train_coco_json, train_image_dir)

# Validate validation data
print("Validating validation data:")
val_valid = validate_coco_format(val_coco_json, val_image_dir)

if train_valid and val_valid:
    print("All validations passed successfully!")
else:
    print("Validation failed.")
