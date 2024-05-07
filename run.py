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
    train_dl, val_dl, test_dl, CONFIG, scaler = data_fn(args.setting)
    n_sample = 50
    if args.test:
        CONFIG["train_config"]["epochs"] = 1
        n_sample = 10

    bb = getattr(backbone, CONFIG["backbone"])
    cn = getattr(conditioner, CONFIG["conditioner"], None)
    df = getattr(diffusion, CONFIG["diffusion"])

    save_folder = os.path.join(
        root_pth,
        args.dataset,
        CONFIG["diffusion"],
        f"{args.setting}_{'t' if args.test else n}",
    )
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(root_pth, args.dataset, "scaler.npy"), "wb") as f:
        np.save(f, scaler)
    torch.save(test_dl, os.path.join(root_pth, args.dataset, "test_dl.pt"))
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
        latent_dim = CONFIG['cn_config']['latent_dim'],
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
        params=params,
        device=device,
        output_pth=save_folder,
        **CONFIG["train_config"],
    )
    trainer.train(train_dl, val_dl)
    results = trainer.test(test_dl, scaler=scaler, n_sample=n_sample)
    print("Finish testing! Begain saving!")

    if os.path.isfile(os.path.join(root_pth, args.dataset, "y_real.npy")):
        print("y_test is exist! only save y_pred")
        with open(os.path.join(save_folder, "y_pred.npy"), "wb") as f:
            np.save(f, results[0])
    else:
        with open(os.path.join(save_folder, "y_pred.npy"), "wb") as f:
            np.save(f, results[0])
        with open(os.path.join(root_pth, args.dataset, "y_real.npy"), "wb") as f:
            np.save(f, results[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mfred")
    parser.add_argument("-n", "--num_train", type=int, default=5)
    parser.add_argument("-s", "--setting", type=str, default="mfred")
    parser.add_argument("-t", "--test", action="store_true")
    args = parser.parse_args()
    n = 1 if args.test else args.num_train
    for i in range(n):
        main(args, i)
