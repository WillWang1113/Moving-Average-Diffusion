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


def calculate_metrics(
    y_pred, y_real, normalize=False, kind="freq", kernel_size: list = []
):
    if len(y_pred.shape) == 4:
        print("Evaluate on highest resolution")
        RMSE = rmse(y_pred, y_real, normalize)
        MAE = mae(y_pred, y_real, normalize)
        CRPS = crps(y_pred, y_real)
        return (RMSE, MAE, CRPS)
    elif len(y_pred.shape) == 5:
        print("Evaluate on multi resolutions")
        assert y_pred.shape[-1] == len(kernel_size)
        all_res_metric = []
        for i in range(y_pred.shape[-1]):
            res_y_pred = y_pred[..., i]
            if i != (len(kernel_size) - 1):
                # DOWNSAMPLE
                assert kind in ["freq", "time"]
                if kind == "time":
                    avgpool = AvgPool1d(kernel_size[i], kernel_size[i])
                    res_y_real = torch.from_numpy(y_real)
                    res_y_real = avgpool(res_y_real.permute(0, 2, 1))
                    # res_y_real = functional.interpolate(res_y_real, size=y_real.shape[1]).permute(0, 2, 1).numpy()
                    res_y_real = res_y_real.numpy()

                    # res_y_pred: [n_sample, bs, ts, dim]
                    res_y_pred = torch.from_numpy(res_y_pred).flatten(end_dim=1)
                    # res_y_pred: [n_sample * bs, ts, dim]
                    res_y_pred = functional.interpolate(
                        res_y_pred.permute(0, 2, 1),
                        size=y_real.shape[1] - kernel_size[i] + 1,
                    )
                    res_y_pred = res_y_pred[..., :: kernel_size[i]].permute(0, 2, 1)
                    assert res_y_pred.shape[1:] == res_y_real.shape[1:]
                    res_y_pred = res_y_pred.reshape((-1, *res_y_real.shape)).numpy()
                else:
                    # TODO: frequency downsample
                    avgpool = AvgPool1d(kernel_size[i], kernel_size[i])
                    res_y_real = torch.from_numpy(y_real)
                    res_y_real = avgpool(res_y_real.permute(0, 2, 1))
                    res_y_real = functional.interpolate(res_y_real, size=y_real.shape[1]).permute(0, 2, 1).numpy()
                    pass
            else:
                res_y_real = y_real
            # fig, ax = plt.subplots()
            # ax.plot(res_y_real[0].flatten())
            # ax.plot(res_y_pred[:,0].squeeze().T, 'k', alpha=0.33)
            # fig.savefig(f'assets/test_{i}.png')
            # plt.close()

            RMSE = rmse(res_y_pred, res_y_real, normalize)
            MAE = mae(res_y_pred, res_y_real, normalize)
            CRPS = crps(res_y_pred, res_y_real)
            print(RMSE)
            all_res_metric.append((RMSE, MAE, CRPS))
        return all_res_metric
    else:
        raise ValueError("wrong y_pred shape")
