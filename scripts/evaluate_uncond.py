from argparse import Namespace
import os
from lightning import seed_everything
import torch
import numpy as np
from src.metrics.discriminative_torch import discriminative_score_metrics
from src.metrics.predictive_metrics import predictive_score_metrics
from src.utils.metrics import context_fid
import pandas as pd

seed_everything(9)
root_save = "/home/user/data2/ICML_camera_ready_test/uncond"
args = Namespace(
    input_size=1,
    device="cuda:0",
)
models = ["DDPM_IN_MA"]
datasets = ["ecg", "exchange_rate", "etth2"]
df = pd.DataFrame()
all_results = []
for d in datasets:
    for m in models:
        y_syn = np.load(
            os.path.join(root_save, f"{d}_24_S", m, "cond_None_dtm_True_syn_0.npy")
        )

        y_real = np.load(
            os.path.join(root_save, f"{d}_24_S", m, "cond_None_dtm_True_real.npy")
        )

        # normalize
        y_syn = (y_syn - np.min(y_syn)) / (np.max(y_syn) - np.min(y_syn))
        y_real = (y_real - np.min(y_real)) / (np.max(y_real) - np.min(y_real))
        assert y_syn.shape == y_real.shape

        ds = discriminative_score_metrics(y_real, y_syn, args)
        ps = predictive_score_metrics(y_real, y_syn)
        cfid = context_fid(
            torch.from_numpy(y_syn),
            torch.from_numpy(y_real),
            torch.load(os.path.join(root_save, f"{d}_24_S", "ae.ckpt")),
        )
        all_results.append((d, m, ds, ps, cfid))


all_results = pd.DataFrame(all_results, columns=["data", "model", "ds", "ps", "cfid"])
print(all_results)
# all_results.to_csv("uncond_seqlen_24_align.csv")
