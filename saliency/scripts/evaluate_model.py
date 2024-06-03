import os
import numpy as np
from skimage import io
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, entropy

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder, filename), as_gray=True)
        if img is not None:
            images.append(img)
    return images

def calculate_auc(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    auc = roc_auc_score(gt, pred)
    return auc

def calculate_nss(pred, gt):
    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    if pred_std == 0:
        return 0
    pred_normalized = (pred - pred_mean) / pred_std
    nss = np.mean(pred_normalized * gt)
    return nss

def calculate_cc(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    cc, _ = pearsonr(pred, gt)
    return cc

def calculate_kld(pred, gt):
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)
    kld = entropy(gt, pred)
    return kld

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
