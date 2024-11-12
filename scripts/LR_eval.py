import argparse
import json
import pandas as pd
import torch
import numpy as np
import os
import logging
from src.utils.filters import get_factors
from src import models
import src.benchmarks
from src.benchmarks.PatchTST import Model

# from src.models.diffusion_pl import MADFreq, MADTime
from lightning import Trainer
from lightning.fabric import seed_everything
from src.utils.sample import plot_fcst, temporal_avg
from src.utils.metrics import MAE, MSE, calculate_metrics, metric
import torch.nn.functional as F

# torch.set_float32_matmul_precision('high')
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


def main(args):
    seed_everything(9)

    # load configs
    root_path = os.path.join(
        args.save_dir, f"{args.dataset}_{args.pred_len}_{args.task}"
    )
    seq_length = args.pred_len

    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))
    
    all_metrics = []
    for i in range(args.num_train):

        with open(f"checkpoints/{args.init_model}_{i}/args.txt", "r") as f:
            model_args = json.load(f)
        model_args = argparse.Namespace(**model_args)
        model_class = getattr(src.benchmarks, model_args.model)
        init_model = model_class.Model(configs=model_args)
        init_model.load_state_dict(
            torch.load(f"checkpoints/{args.init_model}_{i}/checkpoint.pth")
        )

        y_pred, y_real = [], []
        for batch in test_dl:
            pred = init_model(batch["observed_data"])
            pred = F.interpolate(
                pred.permute(0, 2, 1),
                size=batch["future_data"].shape[1],
                mode="linear",
            ).permute(0, 2, 1)
            y_pred.append(pred)
            y_real.append(batch["future_data"])
        y_pred = torch.concat(y_pred).detach().cpu().numpy()
        y_real = torch.concat(y_real).detach().cpu().numpy()
        
        metrics = (MSE(y_pred, y_real), MAE(y_pred, y_real))
        # print(metrics)
        all_metrics.append(metrics)
    all_metrics = np.array(all_metrics)
    print(all_metrics)
    np.save(f"checkpoints/{args.init_model}_{i}/low_res_eval.npy", all_metrics)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument(
        "--save_dir", type=str, default="/home/user/data/FrequencyDiffusion/savings"
    )
    parser.add_argument("--dataset", type=str, default="etth2")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--task", type=str, default="S", choices=["S", "M"])
    parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--kind", type=str, required=True, choices=["freq", "time"])

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)
    # Define overrides with dot notation
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--init_model", type=str, default=None)
    args = parser.parse_args()
    main(args)
