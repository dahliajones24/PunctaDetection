import os
import cv2
import json

def convert_16bit_to_8bit_and_rgb(input_image_dir, output_image_dir, dimensions_file):
    # Create output directory if it does not exist
    os.makedirs(output_image_dir, exist_ok=True)

    image_dimensions = {}
    image_files = sorted([f for f in os.listdir(input_image_dir) if f.endswith('.tif')])

    for image_file in image_files:
        # Process images
        image_path = os.path.join(input_image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Warning: Could not open image: {image_file}")
            continue

        # Get image dimensions
        height, width = image.shape[:2]
        jpg_filename = image_file.replace('.tif', '.jpg')
        image_dimensions[jpg_filename] = {'width': width, 'height': height}

        # Normalize to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Convert grayscale to RGB by replicating the grayscale values across the three channels
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)

        output_image_path = os.path.join(output_image_dir, jpg_filename)
        cv2.imwrite(output_image_path, image_rgb)
        print(f"Converted {image_file} to {output_image_path} in RGB format")

    # Save image dimensions to a JSON file
    with open(dimensions_file, 'w') as json_file:
        json.dump(image_dimensions, json_file, indent=4)

    print(f"Saved image dimensions to {dimensions_file}")

# Example usage
input_image_dir = '/home/djones/puncta_det/data_puncta/data_setup/PSD95'
output_image_dir = '/home/djones/puncta_det/data_puncta/data_setup/rgb_images'
dimensions_file = '/home/djones/puncta_det/data_puncta/data_setup/image_dimensions.json'
convert_16bit_to_8bit_and_rgb(input_image_dir, output_image_dir, dimensions_file)

