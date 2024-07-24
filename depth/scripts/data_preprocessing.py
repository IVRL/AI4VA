import os
import cv2
import json
import pandas as pd

# Define paths
data_dir = './data/images/'
processed_data_dir = './data/processed/'
annotations_file = './data/annotations.json'

# Create the processed data directory if it does not exist
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

# Load annotations
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Initialize lists to store CSV data
csv_data = []

# Process each image and its annotations
for img_info in annotations['images']:
    img_id = img_info['id']
    img_name = img_info['file_name']
    img_path = os.path.join(data_dir, img_name)
    
    # Read and preprocess the image
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Warning: Image {img_name} not found.")
        continue
    
    image = cv2.resize(image, (128, 128)) / 255.0
    
    # Save processed image
    cv2.imwrite(os.path.join(processed_data_dir, img_name), image * 255)
    
    # Find annotations for this image
    for annotation in annotations['annotations']:
        if annotation['image_id'] == img_id:
            category_id = annotation['category_id']
            intradepth = annotation['attributes'].get('Intradepth', 0)  # Default to 0 if not present
            interdepth = annotation['attributes'].get('Interdepth', 0)  # Default to 0 if not present
            
            # Append to CSV data list
            csv_data.append([img_id, category_id, intradepth, interdepth])

# Convert CSV data list to a DataFrame
df = pd.DataFrame(csv_data, columns=['img_id', 'category_id', 'pred_Intradepth', 'pred_Interdepth'])

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(processed_data_dir, 'predictions.csv'), index=False)
