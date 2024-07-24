import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = './data/images/'
depth_dir = './data/depth_maps/'
processed_data_dir = './data/processed/'

# Create directories if they don't exist
os.makedirs(processed_data_dir, exist_ok=True)
train_dir = os.path.join(processed_data_dir, 'train')
val_dir = os.path.join(processed_data_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def preprocess_and_save(image_path, depth_path, save_dir):
    image = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or depth is None:
        print(f"Error reading {image_path} or {depth_path}")
        return
    
    # Example preprocessing: resize and normalize
    image = cv2.resize(image, (128, 128)) / 255.0
    depth = cv2.resize(depth, (128, 128)) / 255.0
    
    # Save processed files
    img_name = os.path.basename(image_path)
    depth_name = os.path.basename(depth_path)
    cv2.imwrite(os.path.join(save_dir, img_name), (image * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, depth_name), (depth * 255).astype(np.uint8))

# Collect all file paths
image_paths = [os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)]
depth_paths = [os.path.join(depth_dir, img_name.replace('image', 'depth')) for img_name in os.listdir(data_dir)]

# Split data into training and validation sets
train_image_paths, val_image_paths, train_depth_paths, val_depth_paths = train_test_split(
    image_paths, depth_paths, test_size=0.2, random_state=42)

# Process and save training data
for img_path, depth_path in zip(train_image_paths, train_depth_paths):
    preprocess_and_save(img_path, depth_path, train_dir)

# Process and save validation data
for img_path, depth_path in zip(val_image_paths, val_depth_paths):
    preprocess_and_save(img_path, depth_path, val_dir)

print("Data preprocessing completed successfully.")
