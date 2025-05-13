import numpy as np
import torch
from src.models.diffusion import BaseDiffusion
from torch.nn import AvgPool1d, functional

import matplotlib.pyplot as plt


def plot_fcst(y_pred: np.ndarray, y_real: np.ndarray, save_name, y_pred_point=None):
    fig, ax = plt.subplots(3, 3, figsize=[8, 6])
    ax = ax.flatten()
    if (y_pred.ndim == 4) and (y_pred.shape[:3] == y_real.shape):
        n_sample = y_pred.shape[0]
        bs = y_pred.shape[1]
        n_series = y_real.shape[-1]
        for k in range(len(ax)):
            choose = np.random.randint(0, bs)
            chn_choose = np.random.randint(0, n_series)
            sample_real = y_real[choose, :, chn_choose]
            sample_pred = y_pred[choose, :, chn_choose, :]
            if y_pred_point is not None:
                sample_pred_point = y_pred_point[choose, :, chn_choose]
            else:
                sample_pred_point = np.median(sample_pred, axis=-1)
            ts = range(len(sample_real))
            ax[k].plot(ts, sample_real, label="real")
            ax[k].plot(ts, sample_pred_point, label='point')
            ax[k].fill_between(
                ts,
                sample_pred[..., 0],
                sample_pred[..., -1],
                # c="black",
                color="orange",
                alpha=0.5, label='90PI'
            )
            # ax[k].plot(sample_pred, c="black", alpha=1 / n_sample)
            ax[k].legend()
            ax[k].set_title(f"sample {choose}, chn {chn_choose}")
    elif (y_pred.shape == y_real.shape):
        bs = y_pred.shape[0]
        n_series = y_real.shape[-1]
        for k in range(len(ax)):
            choose = np.random.randint(0, bs)
            chn_choose = np.random.randint(0, n_series)
            sample_real = y_real[choose, :, chn_choose]
            sample_pred = y_pred[choose, :, chn_choose]

            ts = range(len(sample_real))
            ax[k].plot(ts, sample_real, label="real")
            ax[k].plot(ts, sample_pred, label='gen')
            ax[k].legend()
            ax[k].set_title(f"sample {choose}, chn {chn_choose}")
        
        
    # else:
    #     n_sample = y_pred[0].shape[0]
    #     choose = 99
    #     for k in range(len(ax)):
    #         sample_real = y_real[-k - 1][choose, :, 0]
    #         sample_pred = y_pred[-k - 1][:, choose, :, 0].T
    #         ax[k].plot(sample_real, label="real")
    #         ax[k].plot(sample_pred, c="black", alpha=1 / n_sample)
    #         ax[k].legend()
    #         ax[k].set_title(f"MA kernel size: {kernel_size[-k-1]}")
    #     fig.suptitle(f"sample no. {choose}")
    fig.tight_layout()
    fig.savefig(save_name)


def temporal_avg(y_pred: torch.Tensor, y_real: torch.Tensor, kernel_size, kind):
    all_res_pred, all_res_real = [], []
    for i in range(y_pred.shape[-1]):
        res_y_pred = y_pred[..., i]
        ks = kernel_size[i]
        avgpool = AvgPool1d(ks, ks)
        # res_y_real = torch.from_numpy(y_real)
        res_y_real = avgpool(y_real.permute(0, 2, 1)).permute(0, 2, 1)
        # res_y_real = res_y_real.numpy()
        assert kind in ["freq", "time"]
        if kind == "time":
            # res_y_pred: [n_sample, bs, ts, dim]
            res_y_pred = res_y_pred.flatten(end_dim=1)
            # res_y_pred: [n_sample * bs, ts, dim]
            res_y_pred = functional.interpolate(
                res_y_pred.permute(0, 2, 1),
                size=y_real.shape[1] - kernel_size[i] + 1,
                mode="linear",
            )
            res_y_pred = res_y_pred[..., :: kernel_size[i]].permute(0, 2, 1)
            assert res_y_pred.shape[1:] == res_y_real.shape[1:]
            res_y_pred = res_y_pred.reshape((-1, *res_y_real.shape))
        else:
            res_y_pred = res_y_pred[:, :, ks - 1 :: ks, :]
            assert res_y_pred.shape[2] == res_y_real.shape[1]
        all_res_pred.append(res_y_pred.numpy())
        all_res_real.append(res_y_real.numpy())
    return all_res_pred, all_res_real
