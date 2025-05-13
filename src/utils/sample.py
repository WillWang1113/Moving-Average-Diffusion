import matplotlib.pyplot as plt
import numpy as np


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
        
        
    
    fig.tight_layout()
    fig.savefig(save_name)

