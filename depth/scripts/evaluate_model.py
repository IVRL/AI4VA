import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import cv2
import os

def load_data(data_dir, depth_dir):
    images = []
    depths = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        depth_path = os.path.join(depth_dir, img_name.replace('image', 'depth'))
        
        image = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (128, 128)) / 255.0
        depth = cv2.resize(depth, (128, 128)) / 255.0
        
        images.append(image)
        depths.append(depth)
    
    return np.array(images), np.array(depths)

def evaluate_model(model, X_val, y_val):
    pred_values = model.predict(X_val).flatten()
    true_values = y_val.flatten()
    
    true_ranks = np.argsort(np.argsort(true_values))
    pred_ranks = np.argsort(np.argsort(pred_values))
    
    mse_rank = mean_squared_error(true_ranks, pred_ranks)
    return mse_rank

# Paths to validation data
val_data_dir = './data/images/val/'
val_depth_dir = './data/depth_maps/val/'

# Load the validation data
X_val, y_val = load_data(val_data_dir, val_depth_dir)

# Load the trained model
model = load_model('best_model.h5')

# Evaluate the model
mse_rank = evaluate_model(model, X_val, y_val)

# Save results
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, 'evaluation_metrics.txt'), 'w') as file:
    file.write(f'MSE of Rank: {mse_rank}\n')

print(f'MSE of Rank: {mse_rank}')
