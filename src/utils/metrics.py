import numpy as np
import torch
from torch.nn import AvgPool1d, functional
import matplotlib.pyplot as plt
from neuralforecast.losses.numpy import mae, mqloss, mse

# def crps(y_pred, y_real, sample_weight=None):
#     num_samples = y_pred.shape[0]
#     absolute_error = np.mean(np.abs(y_pred - y_real), axis=0)

#     if num_samples == 1:
#         return np.average(absolute_error, weights=sample_weight)

#     y_pred_sort = np.sort(y_pred, axis=0)
#     b0 = y_pred.mean(axis=0)
#     b1_values = y_pred_sort * np.arange(num_samples).reshape(
#         -1, *[1 for _ in range(len(y_pred.shape) - 1)]
#     )
#     b1 = b1_values.mean(axis=0) / num_samples

#     per_obs_crps = absolute_error + b0 - 2 * b1
#     return np.average(per_obs_crps, weights=sample_weight)


# def mpbl(y_pred, y_real, quantiles=[0.05 * (1 + i) for i in range(19)]):
#     y_pred_q = np.quantile(y_pred, quantiles, axis=0)
#     quantiles = np.array(quantiles).reshape(-1, *[1 for _ in range(y_pred_q.ndim - 1)])
#     error = y_real - y_pred_q  # * n*9
#     first_term = quantiles * error
#     second_term = (quantiles - 1) * error
#     loss = np.maximum(first_term, second_term)
#     return loss.mean()


# def mae(y_pred, y_real, normalize=False):
#     if y_pred.ndim <= 3:
#         y_pred_point = y_pred
#     elif y_pred.ndim == 4:
#         y_pred_point = np.median(y_pred, axis=0)
#     else:
#         raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
#     # y_pred_point = np.median(y_pred, axis=0)
#     scale = np.mean(np.abs(y_real)) if normalize else 1
#     return np.mean(np.abs(y_pred_point - y_real)) / scale


# def mse(y_pred, y_real):
#     if y_pred.ndim <= 3:
#         y_pred_point = y_pred
#     elif y_pred.ndim == 4:
#         y_pred_point = np.median(y_pred, axis=0)
#     else:
#         raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
#     # y_pred_point = np.median(y_pred, axis=0)  
#     return np.mean((y_pred_point - y_real) ** 2)


# def rmse(y_pred, y_real, normalize=False):
#     if y_pred.ndim <= 3:
#         y_pred_point = y_pred
#     elif y_pred.ndim == 4:
#         y_pred_point = np.median(y_pred, axis=0)
#     else:
#         raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
#     MSE = np.mean((y_pred_point - y_real) ** 2)
#     scale = np.mean(np.abs(y_real)) if normalize else 1
#     return np.sqrt(MSE) / scale




def calculate_metrics(
    y_pred,
    y_real,
    quantiles,
    # normalize=False
):
    # print(y_pred.shape)
    # print(y_real.shape)
    if isinstance(y_pred, np.ndarray) and isinstance(y_real, np.ndarray):
        print("Evaluate on highest resolution")
        assert y_pred.ndim == 4
        assert y_pred[0].shape == y_real.shape
        y_pred_point = np.mean(y_pred, axis=0)
        y_pred_q = np.quantile(y_pred, quantiles, axis=0)
        y_pred_q = np.transpose(y_pred_q, (1,2,3,0))

        MAE = mae(y_real, y_pred_point)
        MSE = mse(y_real, y_pred_point)
        MQL = mqloss(y_real, y_pred_q, quantiles=np.array(quantiles))
        # RMSE = rmse(y_pred, y_real, normalize)
        # MAE = mae(y_pred, y_real, normalize)
        # CRPS = crps(y_pred, y_real)
        # MPBL = mpbl(y_pred, y_real)
        return (MAE, MSE, MQL), y_pred_q, y_pred_point
    elif isinstance(y_pred, list) and isinstance(y_real, list):
        print("Evaluate on multi resolutions")
        all_res_metric = []
        all_res_pred_q = []
        all_res_pred_point = []
        
        for i in range(len(y_pred)):
            res_y_pred, res_y_real = y_pred[i], y_real[i]

            assert res_y_pred.ndim == 4
            assert res_y_pred[0].shape == y_real.shape
            res_y_pred_point = np.median(res_y_pred, axis=0)
            res_y_pred_q = np.quantile(res_y_pred, quantiles, axis=0)
            res_y_pred_q = np.transpose(res_y_pred_q, (1,2,3,0))

            MAE = mae(res_y_real, res_y_pred_point)
            MSE = mse(res_y_real, res_y_pred_point)
            MQL = mqloss(res_y_real, res_y_pred_q, quantiles=np.array(quantiles))

            all_res_metric.append((MAE, MSE, MQL))
            all_res_pred_q.append(res_y_pred_q)
            all_res_pred_point.append(all_res_pred_point)


            # RMSE = rmse(res_y_pred, res_y_real, normalize)
            # MAE = mae(res_y_pred, res_y_real, normalize)
            # CRPS = crps(res_y_pred, res_y_real)
            # MPBL = mpbl(res_y_pred, res_y_real)
            # print(RMSE)
            # all_res_metric.append((RMSE, MAE, CRPS, MPBL))
        return all_res_metric, all_res_pred_q, all_res_pred_point
    else:
        raise ValueError("wrong y_pred shape")


# def get_bench_metrics(y_pred, y_real, quantiles):
#     RMSE = rmse(np.median(y_pred, axis=-1), y_real)
#     MAE = mae(np.median(y_pred, axis=-1), y_real)
    
    
#     quantiles = np.array(quantiles).reshape(-1, *[1 for _ in range(y_pred.ndim - 1)])
#     error = y_real - y_pred.transpose(3,0,1,2)  # * n*9
#     first_term = quantiles * error
#     second_term = (quantiles - 1) * error
#     MPBL = np.maximum(first_term, second_term).mean()
#     return (RMSE, MAE, MPBL)
