import numpy as np
import torch
from torch.nn import AvgPool1d, functional
import matplotlib.pyplot as plt


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


def mpbl(y_pred, y_real, quantiles=[0.025 * (1 + i) for i in range(39)]):
    y_pred_q = np.quantile(y_pred, quantiles, axis=0)
    quantiles = np.array(quantiles).reshape(-1, *[1 for _ in range(y_pred_q.ndim - 1)])
    error = y_real - y_pred_q  # * n*9
    first_term = quantiles * error
    second_term = (quantiles - 1) * error
    loss = np.maximum(first_term, second_term)
    return loss.mean()


def mae(y_pred, y_real, normalize=False):
    if y_pred.ndim <= 3:
        y_pred_point = y_pred
    elif y_pred.ndim == 4:
        y_pred_point = np.median(y_pred, axis=0)
    else:
        raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
    # y_pred_point = np.median(y_pred, axis=0)
    scale = np.mean(np.abs(y_real)) if normalize else 1
    return np.mean(np.abs(y_pred_point - y_real)) / scale


def mse(y_pred, y_real):
    if y_pred.ndim <= 3:
        y_pred_point = y_pred
    elif y_pred.ndim == 4:
        y_pred_point = np.median(y_pred, axis=0)
    else:
        raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
    # y_pred_point = np.median(y_pred, axis=0)  
    return np.mean((y_pred_point - y_real) ** 2)


def rmse(y_pred, y_real, normalize=False):
    if y_pred.ndim <= 3:
        y_pred_point = y_pred
    elif y_pred.ndim == 4:
        y_pred_point = np.median(y_pred, axis=0)
    else:
        raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
    MSE = np.mean((y_pred_point - y_real) ** 2)
    scale = np.mean(np.abs(y_real)) if normalize else 1
    return np.sqrt(MSE) / scale


def calculate_metrics(
    y_pred,
    y_real,
    normalize=False,
):
    # print(y_pred.shape)
    # print(y_real.shape)
    if isinstance(y_pred, np.ndarray) and isinstance(y_real, np.ndarray):
        print("Evaluate on highest resolution")
        RMSE = rmse(y_pred, y_real, normalize)
        MAE = mae(y_pred, y_real, normalize)
        CRPS = crps(y_pred, y_real)
        MPBL = mpbl(y_pred, y_real)
        return (RMSE, MAE, CRPS, MPBL)
    elif isinstance(y_pred, list) and isinstance(y_real, list):
        print("Evaluate on multi resolutions")
        all_res_metric = []
        for i in range(len(y_pred)):
            res_y_pred, res_y_real = y_pred[i], y_real[i]

            # fig, ax = plt.subplots()
            # ax.plot(res_y_real[0].flatten())
            # ax.plot(np.median(res_y_pred, axis=0)[0].flatten())
            # # ax.plot(res_y_pred[:,0].squeeze().T, 'k', alpha=0.33)
            # fig.savefig(f'assets/test_{i}.png')
            # plt.close()

            RMSE = rmse(res_y_pred, res_y_real, normalize)
            MAE = mae(res_y_pred, res_y_real, normalize)
            CRPS = crps(res_y_pred, res_y_real)
            MPBL = mpbl(res_y_pred, res_y_real)
            # print(RMSE)
            all_res_metric.append((RMSE, MAE, CRPS, MPBL))
        return all_res_metric
    else:
        raise ValueError("wrong y_pred shape")


def get_bench_metrics(y_pred, y_real, quantiles):
    RMSE = rmse(np.median(y_pred, axis=-1), y_real)
    MAE = mae(np.median(y_pred, axis=-1), y_real)
    
    
    quantiles = np.array(quantiles).reshape(-1, *[1 for _ in range(y_pred.ndim - 1)])
    error = y_real - y_pred.transpose(3,0,1,2)  # * n*9
    first_term = quantiles * error
    second_term = (quantiles - 1) * error
    MPBL = np.maximum(first_term, second_term).mean()
    return (RMSE, MAE, MPBL)



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe