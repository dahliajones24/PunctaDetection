import json
import os
import pandas as pd
from PIL import Image
import shutil

# Paths to your images, CSV data, and output directories
image_directory = '/home/djones/puncta_det/data_puncta/data_setup/new_rgb_bmp_images'
csv_file = '/home/djones/puncta_det/data_puncta/data_setup/bboxess.csv'  # The CSV generated earlier
output_json_path_train = '/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train.json'
output_json_path_val = '/home/djones/puncta_det/data_puncta/puncta/annotations/instances_val.json'
train_image_dir = '/home/djones/puncta_det/data_puncta/puncta/train'
val_image_dir = '/home/djones/puncta_det/data_puncta/puncta/val'

# Ensure output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

# Load your bounding boxes CSV
df = pd.read_csv(csv_file)

# Initialize COCO format dictionaries
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Adding the category (one class "puncta")
coco_format['categories'].append({
    "supercategory": "none",
    "id": 1,
    "name": "puncta"
})

# Iterate over each unique image in the image directory
image_id = 1
annotation_id = 1
image_files = [f for f in os.listdir(image_directory) if f.endswith('.bmp')]

for image_name in image_files:
    image_path = os.path.join(image_directory, image_name)

    # Get image metadata
    with Image.open(image_path) as img:
        width, height = img.size

    # Add image metadata to COCO format
    coco_format['images'].append({
        "file_name": image_name,
        "height": height,
        "width": width,
        "id": image_id
    })

    # Check if this image has any bounding boxes
    image_annotations = df[df['image_id'] == image_name]

    if not image_annotations.empty:
        for _, row in image_annotations.iterrows():
            if pd.notna(row['xmin']):  # Skip rows with no bounding boxes
                # Add each annotation with dummy segmentation
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                segmentation = [
                    [xmin, ymin,  # Bottom left
                     xmax, ymin,  # Bottom right
                     xmax, ymax,  # Top right
                     xmin, ymax]  # Top left
                ]

                coco_format['annotations'].append({
                    "segmentation": segmentation,  # Dummy segmentation based on bounding box
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "category_id": 1,  # Since we have only one category "puncta"
                    "id": annotation_id
                })
                annotation_id += 1

    # Increment image_id for the next image
    image_id += 1

# Split into train and validation (if needed)
train_percentage = 0.8
train_count = int(len(coco_format['images']) * train_percentage)

train_data = {
    "images": coco_format['images'][:train_count],
    "annotations": [ann for ann in coco_format['annotations'] if ann['image_id'] <= train_count],
    "categories": coco_format['categories']
}

val_data = {
    "images": coco_format['images'][train_count:],
    "annotations": [ann for ann in coco_format['annotations'] if ann['image_id'] > train_count],
    "categories": coco_format['categories']
}

# Copy images to the corresponding train/val directories
for i, img in enumerate(coco_format['images']):
    src_image_path = os.path.join(image_directory, img['file_name'])
    if i < train_count:
        dest_image_path = os.path.join(train_image_dir, img['file_name'])
    else:
        dest_image_path = os.path.join(val_image_dir, img['file_name'])

    shutil.copyfile(src_image_path, dest_image_path)

# Save the JSON files
with open(output_json_path_train, 'w') as f:
    json.dump(train_data, f)

with open(output_json_path_val, 'w') as f:
    json.dump(val_data, f)

print(f"COCO formatted JSON files saved to {output_json_path_train} and {output_json_path_val}")
print(f"Images copied to {train_image_dir} and {val_image_dir}")
