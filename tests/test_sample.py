import torch
import numpy as np
import os

from src.train import Sampler
from src.utils.metrics import calculate_metrics


root_path = "/home/user/data/FrequencyDiffusion/savings"
dataset = "mfred"
root_path = os.path.join(root_path, dataset)
exp_dirs = [
    # f"{dataset}_cnn_freq",
    # f"{dataset}_cnn_time",
    # f"{dataset}_mlp_freq",
    # f"{dataset}_mlp_time",
    f"{dataset}_freqlinear_time",
]
num_training = 5
with open(os.path.join(root_path, "scaler.npy"), "rb") as f:
    scaler = np.load(f,allow_pickle='TRUE').item()
test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))


df_out = {d:None for d in exp_dirs}

for d in exp_dirs:
    avg_m = []
    for i in range(num_training):
        read_d = os.path.join(root_path, d + f"_{i}", "diffusion.pt")
        print(read_d)
        diff = torch.load(read_d)
        sampler = Sampler(diff, 100, scaler)
        y_pred, y_real = sampler.sample(test_dl)
        m = calculate_metrics(y_pred, y_real)
        print(m)
        

        