import os
import cv2

def test_rgb_conversion(image_dir):
    # Get a list of all jpg files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    if not image_files:
        print("No JPG images found in the directory.")
        return

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not open image {image_file}")
            continue

        # Check the number of channels in the image
        if len(image.shape) == 3 and image.shape[2] == 3:
            print(f"Success: {image_file} is an RGB image with 3 channels.")
        else:
            print(f"Failure: {image_file} is not an RGB image.")

# Example usage
image_dir = '/home/djones/puncta_det/data_puncta/data_setup/rgb_images'
test_rgb_conversion(image_dir)
