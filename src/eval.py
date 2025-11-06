import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def mae(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float('nan')
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def precision_at_k_matrix(true_matrix, pred_matrix, k=5):
    # For each patient (row), compute Precision@k
    n_p, n_d = true_matrix.shape
    precisions = []
    for i in range(n_p):
        true_row = true_matrix[i]
        pred_row = pred_matrix[i]
        # define positives as top values in true_row among observed
        obs_mask = ~np.isnan(true_row)
        if obs_mask.sum() == 0:
            continue
        # treat true relevance as ranking by true values
        topk_pred_idx = np.argsort(-pred_row)[:k]
        # consider a positive if the true value is above the median of observed
        thr = np.nanmedian(true_row)
        hits = 0
        for idx in topk_pred_idx:
            if not np.isnan(true_row[idx]) and true_row[idx] >= thr:
                hits += 1
        precisions.append(hits / k)
    if len(precisions) == 0:
        return float('nan')
    return float(np.mean(precisions))


def ndcg_at_k(true_matrix, pred_matrix, k=5):
    # Simple NDCG@K where relevance is true value (non-nan)
    import math
    n_p, n_d = true_matrix.shape
    scores = []
    for i in range(n_p):
        true_row = true_matrix[i]
        pred_row = pred_matrix[i]
        obs_mask = ~np.isnan(true_row)
        if obs_mask.sum() == 0:
            continue
        order = np.argsort(-pred_row)[:k]
        dcg = 0.0
        for rank, idx in enumerate(order, start=1):
            rel = 0.0 if np.isnan(true_row[idx]) else float(true_row[idx])
            dcg += (2 ** rel - 1) / math.log2(rank + 1)
        # ideal
        ideal_order = np.argsort(-np.where(np.isnan(true_row), -np.inf, true_row))[:k]
        idcg = 0.0
        for rank, idx in enumerate(ideal_order, start=1):
            rel = 0.0 if np.isnan(true_row[idx]) else float(true_row[idx])
            idcg += (2 ** rel - 1) / math.log2(rank + 1)
        scores.append(dcg / (idcg + 1e-9))
    if len(scores) == 0:
        return float('nan')
    return float(np.mean(scores))
