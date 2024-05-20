import argparse
import os
import random

import numpy as np
import torch

from src.datamodule import dataset
from src.models import backbone, conditioner, diffusion
from src.utils.train import Trainer
import json
from src.models.benchmarks import DLinear


root_pth = "/home/user/data/FrequencyDiffusion/savings"
fix_seed = 9
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args, n):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)

    data_fn = getattr(dataset, args.dataset)
    train_dl, val_dl, test_dl, CONFIG, scaler = data_fn(args.setting)
    
    batch = next(iter(train_dl))
    seq_length, seq_channels = (
        batch["observed_data"].shape[1],
        batch["observed_data"].shape[2],
    )
    target_seq_length, target_seq_channels = (
        batch["future_data"].shape[1],
        batch["future_data"].shape[2],
    )
    future_seq_length, future_seq_channels = (
        batch["future_features"].shape[1],
        batch["future_features"].shape[2],
    )
    model = DLinear(
        seq_channels=seq_channels,
        seq_length=seq_length,
        future_seq_channels=future_seq_channels,
        future_seq_length=future_seq_length,
    ).to(device)
    
    exp_name = f"{model._get_name()}_{'t' if args.test else n}"

    save_folder = os.path.join(
        root_pth,
        args.dataset,
        'benchmarks',
        exp_name,
    )
    os.makedirs(save_folder, exist_ok=True)
    
    trainer = Trainer(
        diffusion=model,
        device=device,
        output_pth=save_folder,
        exp_name=exp_name,
        **CONFIG["train_config"],
    )
    trainer.train(train_dl, val_dl)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mfred")
    parser.add_argument("-n", "--num_train", type=int, default=5)
    parser.add_argument("-s", "--setting", type=str, default="default_mfred")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)


    args = parser.parse_args()
    n = 1 if args.test else args.num_train
    for i in range(n):
        main(args, i)
