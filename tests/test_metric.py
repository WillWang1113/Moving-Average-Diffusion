import random
import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils.metrics import calculate_metrics
import pandas as pd

root_path = "/home/user/data/FrequencyDiffusion/savings"
dataset = "mfred"
root_path = os.path.join(root_path, dataset)
with open(os.path.join(root_path, "y_real.npy"), "rb") as f:
    y_real = np.load(f)
exp_dirs = [
    # f"{dataset}_cnn_freq",
    # f"{dataset}_cnn_time",
    f"{dataset}_mlp_freq",
    f"{dataset}_mlp_time",
    f"{dataset}_freqlinear_time",
]
num_training = 5
# exp_dirs = os.listdir(root_path)
# exp_dirs.sort()

df_out = {d:None for d in exp_dirs}

for d in exp_dirs:
    print(d)
    avg_m = []
    for i in range(num_training):
        # i = 't'
        read_d = os.path.join(root_path, d + f"_{i}", "y_pred.npy")
        with open(read_d, "rb") as f:
            y_pred = np.load(f)

        # y_pred = np.transpose(y_pred, (1,2,3,0))
        m = calculate_metrics(y_pred, y_real)
        avg_m.append(m)    
        n_sample, bs, ts, dims = y_pred.shape

        fig, ax = plt.subplots(2,2)
        ax = ax.flatten()
        for k in range(4):
            np.random.seed(k)
            random.seed(k)
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
# df_out.to_csv(os.path.join(root_path, 'metrics.csv'))
        


