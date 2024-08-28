import os
import cv2
import json
import pandas as pd

def convert_16bit_to_8bit_and_rgb(input_image_dir, output_image_dir):
    # Create output directory if it does not exist
    os.makedirs(output_image_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_image_dir) if f.endswith('.tif')])

    for image_file in image_files:
        # Process images
        image_path = os.path.join(input_image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Warning: Could not open image: {image_file}")
            continue

        # Normalize to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Convert grayscale to RGB by replicating the grayscale values across the three channels
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)

        # Save the RGB image as a jpg file
        jpg_filename = image_file.replace('.tif', '.jpg')
        output_image_path = os.path.join(output_image_dir, jpg_filename)
        cv2.imwrite(output_image_path, image_rgb)
        print(f"Converted {image_file} to {output_image_path} in RGB format")

# Example usage
input_image_dir = '/home/djones/punctadetection/data/data_setup/PSD95'
output_image_dir = '/home/djones/punctadetection/data/data_setup/rgb_images'
convert_16bit_to_8bit_and_rgb(input_image_dir, output_image_dir)
