import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, entropy

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
