import os
import numpy as np
from skimage import io
from metrics import calculate_auc, calculate_nss, calculate_cc, calculate_kld

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder, filename), as_gray=True)
        if img is not None:
            images.append(img)
    return images

def evaluate_model(predictions_folder, gt_folder):
    predictions = load_images_from_folder(predictions_folder)
    ground_truths = load_images_from_folder(gt_folder)

    assert len(predictions) == len(ground_truths), "Number of prediction and ground truth images must be the same."

    auc_scores = []
    nss_scores = []
    cc_scores = []
    kld_scores = []

    for pred, gt in zip(predictions, ground_truths):
        auc_scores.append(calculate_auc(pred, gt))
        nss_scores.append(calculate_nss(pred, gt))
        cc_scores.append(calculate_cc(pred, gt))
        kld_scores.append(calculate_kld(pred, gt))

    metrics = {
        'AUC': np.mean(auc_scores),
        'NSS': np.mean(nss_scores),
        'CC': np.mean(cc_scores),
        'KLD': np.mean(kld_scores)
    }

    return metrics

if __name__ == "__main__":
    predictions_folder = "path/to/predictions_folder"
    gt_folder = "path/to/gt_folder"

    metrics = evaluate_model(predictions_folder, gt_folder)
    print(f"Model Evaluation Metrics:\nAUC: {metrics['AUC']}\nNSS: {metrics['NSS']}\nCC: {metrics['CC']}\nKLD: {metrics['KLD']}")
