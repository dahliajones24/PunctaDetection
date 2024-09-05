import pandas as pd
import os
import math
from PIL import Image
import shutil  # For copying files

# Define the directory paths
excel_directory = '/home/djones/puncta_det/data_puncta/data_setup/groundtruth'
image_directory = '/home/djones/puncta_det/data_puncta/data_setup/rgb_bmp_images'  # Original image folder path
output_image_directory = '/home/djones/puncta_det/data_puncta/data_setup/new_rgb_bmp_images'  # New folder for filtered images
output_csv_path = '/home/djones/puncta_det/data_puncta/data_setup/bboxess.csv'  # Path to save the consolidated CSV file

# Create the new directory if it doesn't exist
os.makedirs(output_image_directory, exist_ok=True)

# Initialize an empty list to store the bounding box data
all_bboxes = []

# Iterate over each Excel file in the directory
for excel_file in os.listdir(excel_directory):
    if excel_file.endswith('.xlsx'):
        # Construct the full path to the Excel file
        file_path = os.path.join(excel_directory, excel_file)

        # Load the Excel file
        df = pd.read_excel(file_path, header=None)

        # Determine the corresponding image file name
        image_name = excel_file.replace('.xlsx', '.bmp')  # Ensure correct file extension
        image_path = os.path.join(image_directory, image_name)

        # Check if the Excel file is empty or malformed
        if df.empty or df.shape[1] < 2:
            print(f"Removing {excel_file} and skipping {image_name} due to empty or malformed Excel file.")
            # Remove the Excel file
            os.remove(file_path)
            continue  # Skip processing this file

        # Get the image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Rename columns to 'x' and 'y'
        df.columns = ['x', 'y']

        # Define bounding box size (11x11 pixels, so half is 5.5)
        half_bbox_size = 5.5

        # Calculate xmin, ymin, xmax, ymax
        df['xmin'] = df['x'] - half_bbox_size
        df['ymin'] = df['y'] - half_bbox_size
        df['xmax'] = df['x'] + half_bbox_size
        df['ymax'] = df['y'] + half_bbox_size

        # Clip values to stay within the image dimensions
        df['xmin'] = df['xmin'].clip(lower=0)
        df['ymin'] = df['ymin'].clip(lower=0)
        df['xmax'] = df['xmax'].clip(upper=image_width)
        df['ymax'] = df['ymax'].clip(upper=image_height)

        # Round or floor the coordinates to ensure they're integers
        df['xmin'] = df['xmin'].apply(math.floor)
        df['ymin'] = df['ymin'].apply(math.floor)
        df['xmax'] = df['xmax'].apply(math.ceil)
        df['ymax'] = df['ymax'].apply(math.ceil)

        # Add the class name and image ID
        df['class'] = 'puncta'
        df['image_id'] = image_name

        # Append the processed DataFrame to the list
        all_bboxes.append(df[['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class']])

        # Copy the valid image to the new directory
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(output_image_directory, image_name))
            print(f"Copied {image_name} to {output_image_directory}")

# Concatenate all bounding box data into a single DataFrame
if all_bboxes:
    final_df = pd.concat(all_bboxes, ignore_index=True)

    # Save the result to a single CSV file
    final_df.to_csv(output_csv_path, index=False)
    print(f"Bounding boxes saved to {output_csv_path}")
else:
    print("No valid bounding boxes found. CSV file not created.")
