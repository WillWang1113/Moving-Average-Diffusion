import argparse
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml

from src.datamodule import dataset
from src.models import backbone, conditioner, diffusion
from src.utils.fourier import idft
from src.utils.schedule import linear_schedule, variance_schedule
from src.utils.train import Trainer, get_expname_
import json


# root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings"
root_pth = "/home/user/data/FrequencyDiffusion/savings"
fix_seed = 9
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config, run_args, n):
    device = f"cuda:{run_args['gpu']}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)

    data_fn = getattr(dataset, run_args["dataset"])
    if run_args["test"]:
        config["train_config"]["epochs"] = 3

    bb_, cn_, df_ = (
        config["bb_config"].get("name"),
        config["cn_config"].get("name"),
        config["diff_config"].pop("name"),
    )
    train_dl, val_dl, test_dl, scaler = data_fn()

    df = getattr(diffusion, df_)
    exp_name = f"test_{get_expname_(config, bb_, cn_, df_)}{'t' if run_args['test'] else n}"
    # exp_name = f"{bb_}_{CONFIG['diff_config']}_{'t' if args.test else n}"
    print(exp_name)
    save_folder = os.path.join(
        root_pth,
        run_args["dataset"],
        df_,
        exp_name,
    )
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(config, w, indent=2)

    with open(os.path.join(root_pth, run_args["dataset"], "scaler.npy"), "wb") as f:
        np.save(f, scaler)
    torch.save(test_dl, os.path.join(root_pth, run_args["dataset"], "test_dl.pt"))
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

    

    config["bb_config"]["seq_channels"] = target_seq_channels
    config["bb_config"]["seq_length"] = target_seq_length
    config["bb_config"]["latent_dim"] = config["cn_config"]["latent_dim"]

    config["cn_config"]["seq_channels"] = seq_channels
    config["cn_config"]["seq_length"] = seq_length
    config["cn_config"]["future_seq_channels"] = future_seq_channels
    config["cn_config"]["future_seq_length"] = future_seq_length

    diff = df(
        backbone_config=config["bb_config"],
        conditioner_config=config["cn_config"],
        **config["diff_config"],
    )
    # print("\n")
    # print("MODEL PARAM:")
    # print(sum([p.numel() for p in diff.parameters()]))
    # return 0
    trainer = Trainer(
        smoke_test=run_args["test"],
        device=device,
        output_pth=save_folder,
        exp_name=exp_name,
        **config["train_config"],
    )
    # trainer.train(train_dl)
    trainer.fit(diff, train_dl, val_dl)
    diff.config_sampling()
    preds = trainer.predict(diff, test_dl)
    print(preds[0][0][0].shape)
    # plt.plot(preds[0][0][0].flatten())
    plt.plot(idft(preds[0][0][[0]], real_imag=True).flatten())
    plt.savefig('test_new.png')

if __name__ == "__main__":
    # FOR MY OWN EXPERIMENTS
    args = {"dataset": "mfred", "num_train": 1, "test": False, "gpu": 0}
    config = yaml.safe_load(open("configs/MAD.yaml", "r"))
    main(config, args, 0)

