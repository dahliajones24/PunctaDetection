import os
import json

def validate_coco_structure(json_path, expected_num_images, labeled=True):
    """
    Validates the structure of the COCO JSON file to ensure correctness.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validate sections
    assert 'images' in data, f"Missing 'images' section in {json_path}"
    assert 'annotations' in data, f"Missing 'annotations' section in {json_path}"
    assert 'categories' in data, f"Missing 'categories' section in {json_path}"

    # Check the number of images
    num_images = len(data['images'])
    assert num_images == expected_num_images, f"Expected {expected_num_images} images, found {num_images} in {json_path}"

    # If labeled, check for annotations
    if labeled:
        image_ids_with_annotations = {ann['image_id'] for ann in data['annotations']}
        assert len(image_ids_with_annotations) > 0, f"No annotations found in labeled set {json_path}"
        # Ensure that there are no images without corresponding annotations if labeled
        for img in data['images']:
            if img['id'] in image_ids_with_annotations:
                assert any(ann['image_id'] == img['id'] for ann in data['annotations']), f"Image {img['id']} has no corresponding annotation in {json_path}"
    else:
        assert len(data['annotations']) == 0, f"Unexpected annotations found in unlabeled set {json_path}"

    print(f"Validation passed for {'labeled' if labeled else 'unlabeled'} set: {json_path}")

def main():
    out_root = '/home/djones/punctadetection/data/active_learning'  # Update with your path
    n_splits = 3
    n_labeled = 100  # Number of labeled images per split

    # Read the total images from the training dataset JSON to calculate correctly
    with open('/home/djones/punctadetection/data/puncta/annotations/instances_train.json', 'r') as f:  # Update the path to your training JSON
        training_data = json.load(f)
        total_images_in_training = len(training_data['images'])

    for i in range(1, n_splits + 1):
        labeled_json = os.path.join(out_root, f'puncta_{n_labeled}_{i}_labeled.json')
        unlabeled_json = os.path.join(out_root, f'puncta_{n_labeled}_{i}_unlabeled.json')

        # Validate the labeled set
        validate_coco_structure(labeled_json, n_labeled, labeled=True)

        # Validate the unlabeled set
        unlabeled_count = total_images_in_training - n_labeled  # Correctly calculate remaining unlabeled images
        validate_coco_structure(unlabeled_json, unlabeled_count, labeled=False)

if __name__ == '__main__':
    main()
