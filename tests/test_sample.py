import pandas as pd
import torch
import numpy as np
import os

from src.train import Sampler
from src.utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

root_path = "/home/user/data/FrequencyDiffusion/savings/mfred"
dataset = "MovingAvgDiffusion"
exp_path = os.path.join(root_path, dataset)
with open(os.path.join(root_path, "y_real.npy"), "rb") as f:
    y_real = np.load(f)
exp_dirs = [
    # "cnn_freq",
    "cnn_time",
    # "mlp_freq",
    # "cmlp_freq",
    "mlp_time",
    # "freqlinear_time",
]
num_training = 5

with open(os.path.join(root_path, "scaler.npy"), "rb") as f:
    scaler = np.load(f,allow_pickle='TRUE').item()
test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))


df_out = {d:None for d in exp_dirs}

for d in exp_dirs:
    avg_m = []
    for i in range(num_training):
        read_d = os.path.join(exp_path, d + f"_{i}", "diffusion.pt")
        print(read_d)
        diff = torch.load(read_d)
        # diff.fast_sample = True
        sampler = Sampler(diff, 50, scaler)
        y_pred, y_real = sampler.sample(test_dl)
        m = calculate_metrics(y_pred, y_real)
        n_sample, bs, ts, dims = y_pred.shape
        
        print(m)
        
        if i == 0 or i == 't':
            fig, ax = plt.subplots(2,2)
            ax = ax.flatten()
            for k in range(4):
                np.random.seed(k)
                choose = np.random.randint(0, bs)
                sample_real = y_real[choose,:,0]
                sample_pred = y_pred[:, choose, :, 0].T
                # ax[k].scatter(sample_pred[:,2], sample_pred[:,5])
                ax[k].plot(sample_real, label="real")
                ax[k].legend()
                ax[k].plot(sample_pred, c="black", alpha=1 / n_sample)
            # fig.suptitle(f"ma_term: {L}")
            fig.tight_layout()
            fig.savefig(f"assets/{d}_{i}.png")
    avg_m = np.array(avg_m)
    df_out[d] = avg_m.mean(axis=0)
df_out = pd.DataFrame(df_out, index=['RMSE','MAE','CRPS']).T
print(df_out)
df_out.to_csv('stochatic metrics.csv')
# df_out.to_csv(os.path.join(root_path, 'metrics.csv'))
        

        