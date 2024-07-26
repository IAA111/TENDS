import numpy as np
import torch
# 对 nan 用邻居值的均值补全
def fillna_mean_of_neighbors(X):
    for i in range(X.shape[1]):
        arr = X[:, i]
        new_col = np.zeros_like(arr)

        for k in range(len(arr)):
            if np.isnan(arr[k]):
                upper_neighbor = np.nan
                lower_neighbor = np.nan

                for idx in range(k - 1, -1, -1):
                    if not np.isnan(arr[idx]):
                        upper_neighbor = arr[idx]
                        break

                for idx in range(k + 1, len(arr), 1):
                    if not np.isnan(arr[idx]):
                        lower_neighbor = arr[idx]
                        break

                if np.isnan(upper_neighbor):
                    new_col[k] = lower_neighbor
                elif np.isnan(lower_neighbor):
                    new_col[k] = upper_neighbor
                else:
                    new_col[k] = (upper_neighbor + lower_neighbor) / 2
            else:
                new_col[k] = arr[k]

        X[:, i] = new_col
    return X


def nanmean(v, *args, **kwargs):
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def quantile(X, q, dim=None):
    return X.kthvalue(int(q * len(X)), dim=dim)[0]

def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult

def MAE(X ,X_true, mask):
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()

#  Root Mean Squared Error (MAE) between imputed variables and ground truth : X imputed X_true: Ground truth
def RMSE(X ,X_true, mask):
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else:  # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())