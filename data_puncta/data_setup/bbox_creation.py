import pandas as pd
import os
import math
from PIL import Image

# Define the directory paths
excel_directory = '/home/djones/puncta_det/data_puncta/data_setup/groundtruth'
image_directory = '/home/djones/puncta_det/data_puncta/data_setup/rgb_images'  # Replace with your actual image folder path
output_csv_path = '/home/djones/puncta_det/data_puncta/data_setup/bboxes.csv'  # Path to save the consolidated CSV file

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
        image_name = excel_file.replace('.xlsx', '.jpg')
        image_path = os.path.join(image_directory, image_name)

        # Get the image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Handle empty or malformed Excel files
        if df.empty or df.shape[1] < 2:
            print(f"Warning: {excel_file} is empty or malformed. Including {image_name} with no bounding boxes.")
            # Create a DataFrame for the image with no bounding boxes
            empty_df = pd.DataFrame({
                'image_id': [image_name],
                'xmin': [None],
                'ymin': [None],
                'xmax': [None],
                'ymax': [None],
                'class': [None]
            })
            all_bboxes.append(empty_df)
            continue

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

# Concatenate all bounding box data into a single DataFrame
final_df = pd.concat(all_bboxes, ignore_index=True)

# Save the result to a single CSV file
final_df.to_csv(output_csv_path, index=False)

print(f"Bounding boxes saved to {output_csv_path}")
