import os
import cv2
import pandas as pd
import json

def test_bboxes_and_images(output_image_dir, output_bbox_dir, dimensions_file):
    # Load image dimensions from the JSON file
    with open(dimensions_file, 'r') as json_file:
        image_dimensions = json.load(json_file)

    image_files = sorted([f for f in os.listdir(output_image_dir) if f.endswith('.jpg')])
    bbox_files = sorted([f for f in os.listdir(output_bbox_dir) if f.endswith('.csv')])

    # Test 1: Ensure each image has a corresponding bounding box file
    assert len(image_files) == len(bbox_files), "Mismatch between number of images and number of bounding box files."

    total_bboxes = 0
    all_bboxes_valid = True

    for image_file, bbox_file in zip(image_files, bbox_files):
        # Verify corresponding image and bbox files
        assert image_file.replace('.jpg', '.csv') == bbox_file, f"Mismatch: {image_file} and {bbox_file}"

        # Load image and check it's in RGB format
        image_path = os.path.join(output_image_dir, image_file)
        image = cv2.imread(image_path)

        assert image is not None, f"Image {image_file} could not be loaded."
        assert image.shape[2] == 3, f"Image {image_file} is not in RGB format."

        height, width = image.shape[:2]
        assert image_dimensions[image_file]['width'] == width, f"Width mismatch for {image_file}"
        assert image_dimensions[image_file]['height'] == height, f"Height mismatch for {image_file}"

        # Load bounding box CSV and validate against image dimensions
        bbox_path = os.path.join(output_bbox_dir, bbox_file)
        bbox_data = pd.read_csv(bbox_path)

        for _, bbox in bbox_data.iterrows():
            x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
            if not (0 <= x_min < x_max <= width) or not (0 <= y_min < y_max <= height):
                print(f"Invalid bounding box in {bbox_file}: [{x_min}, {y_min}, {x_max}, {y_max}]")
                all_bboxes_valid = False

        total_bboxes += len(bbox_data)

    print(f"Total number of images: {len(image_files)}")
    print(f"Total number of bounding boxes: {total_bboxes}")

    if all_bboxes_valid:
        print("All bounding boxes are valid.")
    else:
        print("Some bounding boxes were invalid.")

    print("All tests were successful." if all_bboxes_valid else "Some tests failed.")

# Example usage
output_image_dir = '/home/djones/puncta_det/data_puncta/data_setup/rgb_images'
output_bbox_dir = '/home/djones/puncta_det/data_puncta/data_setup/bboxes'
dimensions_file = '/home/djones/puncta_det/data_puncta/data_setup/image_dimensions.json'

test_bboxes_and_images(output_image_dir, output_bbox_dir, dimensions_file)





