import os
import cv2

data_dir = './data/images/'
depth_dir = './data/depth_maps/'
processed_data_dir = './data/processed/'

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    depth_path = os.path.join(depth_dir, img_name.replace('image', 'depth'))
    
    image = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    # Example preprocessing: resize and normalize
    image = cv2.resize(image, (128, 128)) / 255.0
    depth = cv2.resize(depth, (128, 128)) / 255.0
    
    # Save processed files
    cv2.imwrite(os.path.join(processed_data_dir, img_name), image * 255)
    cv2.imwrite(os.path.join(processed_data_dir, img_name.replace('image', 'depth')), depth * 255)
