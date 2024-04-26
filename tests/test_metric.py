import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from src.utils.fourier import idft
from src.utils.filters import get_factors

root_path = '/home/user/data/FrequencyDiffusion/savings'
dataset = 'mfred'
root_path = os.path.join(root_path, dataset)
with open(os.path.join(root_path, 'y_real.npy'), 'rb') as f:
    y_real = np.load(f)
exp_dirs = os.listdir(root_path)
exp_dirs.sort()

for d in exp_dirs:
    read_d = os.path.join(root_path, d, 'y_pred.npy')
    with open(read_d, 'rb') as f:
        y_pred = np.load(f)
    y_pred = np.transpose(y_pred, (1,2,3,0))
    print(y_pred.shape)


    bs, ts, dims, n_sample = y_pred.shape
    
    ma_terms = [1] + get_factors(ts) + [ts]
    # assert Ts == len(ma_terms)
    # for ii in range(Ts):
        # if ii == Ts-1:
    # L = ma_terms[-ii - 1]
    sample_real = y_real[0,:,0]
    sample_pred = y_pred[0, :, 0, :]
    # sample_pred = y_pred[:, 0, :, 0, ii]
    print(sample_real.shape)
    print(sample_pred.shape)
    fig, ax = plt.subplots()
    ax.plot(sample_real, label="real")
    ax.legend()
    ax.plot(sample_pred, c="black", alpha=5 / n_sample)
    # fig.suptitle(f"ma_term: {L}")
    fig.tight_layout()
    fig.savefig(f"assets/{d}.png")
        
        # plt.close()
    