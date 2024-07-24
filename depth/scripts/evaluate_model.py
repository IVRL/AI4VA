import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import os
import pandas as pd
import json

def evaluate_model(predictions, ground_truth):
    pred_inter_depths = []
    pred_intra_depths = []
    true_inter_depths = []
    true_intra_depths = []

    # Process each prediction
    for index, row in predictions.iterrows():
        img_id = str(row['img_id'])
        category_id = str(row['category_id'])

        pred_inter_depths.append(row['pred_Interdepth'])
        pred_intra_depths.append(row['pred_Intradepth'])

        if img_id in ground_truth:
            if category_id in ground_truth[img_id]:
                true_inter_depths.append(ground_truth[img_id][category_id]['Interdepth'])
                true_intra_depths.append(ground_truth[img_id][category_id]['Intradepth'])
            else:
                print(f"Warning: Category ID {category_id} not found in ground truth for image {img_id}")
        else:
            print(f"Warning: Image ID {img_id} not found in ground truth")

    # Calculate MSE for inter-depth
    mse_inter = mean_squared_error(true_inter_depths, pred_inter_depths)
    # Calculate MSE for intra-depth
    mse_intra = mean_squared_error(true_intra_depths, pred_intra_depths)

    # Calculate overall MSE
    mse_overall = (mse_inter + mse_intra) / 2

    return mse_inter, mse_intra, mse_overall

# Paths to validation data
val_data_dir = './data/images/val/'
gt_path = './data/annotations.json'  # path to the ground truth annotations

# Load the ground truth annotations
with open(gt_path, 'r') as f:
    ground_truth_data = json.load(f)

# Transform ground truth data for easier lookup
ground_truth = {}
for ann in ground_truth_data['annotations']:
    img_id = str(ann['image_id'])
    category_id = str(ann['category_id'])
    intradepth = ann['attributes'].get('Intradepth', 0)
    interdepth = ann['attributes'].get('Interdepth', 0)
    if img_id not in ground_truth:
        ground_truth[img_id] = {}
    ground_truth[img_id][category_id] = {
        'Intradepth': intradepth,
        'Interdepth': interdepth
    }

# Load the trained model
model = load_model('best_model.h5')

# Load validation images
X_val = []
img_ids = []
for img_info in ground_truth_data['images']:
    img_id = img_info['id']
    img_name = img_info['file_name']
    img_path = os.path.join(val_data_dir, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.resize(image, (128, 128)) / 255.0
        X_val.append(image)
        img_ids.append(img_id)

X_val = np.array(X_val)

# Get predictions
predictions = model.predict(X_val)

# Prepare predictions dataframe
predictions_df = pd.DataFrame(predictions, columns=['pred_Intradepth', 'pred_Interdepth'])
predictions_df['img_id'] = img_ids
predictions_df['category_id'] = [cat_id for img in ground_truth.values() for cat_id in img.keys()]

# Evaluate the model
mse_inter, mse_intra, mse_overall = evaluate_model(predictions_df, ground_truth)

# Save results
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, 'evaluation_metrics.txt'), 'w') as file:
    file.write(f'MSE of Inter-depth: {mse_inter}\n')
    file.write(f'MSE of Intra-depth: {mse_intra}\n')
    file.write(f'Overall MSE: {mse_overall}\n')

print(f'MSE of Inter-depth: {mse_inter}')
print(f'MSE of Intra-depth: {mse_intra}')
print(f'Overall MSE: {mse_overall}')
