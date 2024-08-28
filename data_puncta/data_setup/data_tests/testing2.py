import os
import json

def check_coco_file(json_file, image_dir, labeled=True):
    try:
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return False
    
    # Check for required keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            print(f"{json_file} is missing the '{key}' key.")
            return False
    
    # Check categories
    if len(coco_data['categories']) == 0 or coco_data['categories'][0]['name'] != "puncta":
        print(f"{json_file} has incorrect categories.")
        return False
    
    # Check images
    for image_info in coco_data['images']:
        image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            print(f"Image {image_info['file_name']} in {json_file} is missing from {image_dir}.")
            return False
    
    # Check annotations for labeled data
    if labeled:
        if len(coco_data['annotations']) == 0:
            print(f"No annotations found in labeled file {json_file}.")
            return False
        
        # Ensure each annotation has necessary fields
        for annotation in coco_data['annotations']:
            if 'bbox' not in annotation or len(annotation['bbox']) != 4:
                print(f"Invalid bounding box found in {json_file}.")
                return False
            if 'image_id' not in annotation:
                print(f"Missing image_id in annotation in {json_file}.")
                return False
    else:
        # Ensure unlabeled data has no annotations or an empty annotations array
        if len(coco_data['annotations']) > 0:
            print(f"Unlabeled file {json_file} contains annotations.")
            return False

    print(f"{json_file} passed all checks.")
    return True

# Define paths
coco_dir = "/home/djones/puncta_det/data_puncta/active_learning/"
labeled_files = [
    os.path.join(coco_dir, "coco_2365_labeled_1.json"),
    os.path.join(coco_dir, "coco_2365_labeled_2.json"),
    os.path.join(coco_dir, "coco_2365_labeled_3.json"),
]
unlabeled_files = [
    os.path.join(coco_dir, "coco_2365_unlabeled_1.json"),
    os.path.join(coco_dir, "coco_2365_unlabeled_2.json"),
    os.path.join(coco_dir, "coco_2365_unlabeled_3.json"),
]
image_dir = "/home/djones/puncta_det/data_puncta/puncta/train_image_dir/"

# Run checks
all_checks_passed = True

for labeled_file in labeled_files:
    if not check_coco_file(labeled_file, image_dir, labeled=True):
        all_checks_passed = False

for unlabeled_file in unlabeled_files:
    if not check_coco_file(unlabeled_file, image_dir, labeled=False):
        all_checks_passed = False

if all_checks_passed:
    print("All checks passed successfully!")
else:
    print("Some checks failed. Please review the output for details.")
