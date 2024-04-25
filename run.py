import argparse
import os
import random

import numpy as np
import torch

from src.datamodule import dataset
from src.models import backbone, conditioner, diffusion
from src.train import Trainer


device = "cuda" if torch.cuda.is_available() else "cpu"
root_pth = "/home/user/data/FrequencyDiffusion/savings"
fix_seed = 9
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Using: ", device)


def main(args, n):
    data_fn = getattr(dataset, args.dataset)
    train_dl, val_dl, test_dl, CONFIG, ct = data_fn(args.setting)

    bb = getattr(backbone, CONFIG["backbone"])
    cn = getattr(conditioner, CONFIG["conditioner"], None)
    df = getattr(diffusion, CONFIG["diffusion"])

    save_folder = os.path.join(
        root_pth,
        args.dataset,
        f"{args.setting}_{n}",
    )
    os.makedirs(save_folder, exist_ok=True)

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

    print("\n")
    print("MODEL PARAM:")

    bb_net = bb(
        seq_channels=target_seq_channels,
        seq_length=target_seq_length,
        **CONFIG["bb_config"],
    )
    print("Denoise Network:\t", sum([torch.numel(p) for p in bb_net.parameters()]))
    params = list(bb_net.parameters())

    if cn is not None:
        cond_net = cn(
            seq_channels=seq_channels,
            seq_length=seq_length,
            future_seq_channels=future_seq_channels,
            future_seq_length=future_seq_length,
            **CONFIG["cn_config"],
        )
        print(
            "Condition Network:\t", sum([torch.numel(p) for p in cond_net.parameters()])
        )
        params = params + list(cond_net.parameters())
    else:
        cond_net = None
    print("\n")

    diff = df(
        backbone=bb_net,
        conditioner=cond_net,
        device=device,
        **CONFIG["diff_config"],
    )

    trainer = Trainer(
        diffusion=diff,
        optimizer=torch.optim.Adam(params, lr=CONFIG["train_config"]["lr"]),
        device=device,
        output_pth=save_folder,
        **CONFIG["train_config"],
    )
    trainer.train(train_dl, val_dl)
    trainer.train(val_dl, epochs=10)
    results = trainer.test(test_dl)
    torch.save((results, ct), os.path.join(save_folder, "results.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mfred")
    parser.add_argument("-n", "--num_train", type=int, default=5)
    parser.add_argument("-s", "--setting", type=str, default="mfred")
    args = parser.parse_args()

    for i in range(args.num_train):
        main(args, i)
