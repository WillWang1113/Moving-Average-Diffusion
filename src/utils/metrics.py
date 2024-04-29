import numpy as np


def crps(y_pred, y_real, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_real), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred_sort = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred_sort * np.arange(num_samples).reshape(
        -1, *[1 for _ in range(len(y_pred.shape) - 1)]
    )
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.average(per_obs_crps, weights=sample_weight)


def mae(y_pred, y_real, normalize=False):
    y_pred_point = np.median(y_pred, axis=0)
    scale = np.mean(np.abs(y_real)) if normalize else 1
    return np.mean(np.abs(y_pred_point - y_real)) / scale


def mse(y_pred, y_real):
    y_pred_point = np.median(y_pred, axis=0)
    return np.mean((y_pred_point - y_real) ** 2)


def rmse(y_pred, y_real, normalize=False):
    y_pred_point = np.median(y_pred, axis=0)
    MSE = np.mean((y_pred_point - y_real) ** 2)
    scale = np.mean(np.abs(y_real)) if normalize else 1
    return np.sqrt(MSE) / scale


def calculate_metrics(y_pred, y_real, normalize=False):
    NRMSE = rmse(y_pred, y_real, normalize)
    MAE = mae(y_pred, y_real, normalize)
    CRPS = crps(y_pred, y_real)
    return (NRMSE, MAE, CRPS)
