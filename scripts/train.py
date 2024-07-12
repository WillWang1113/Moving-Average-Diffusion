import argparse
import os

import numpy as np
import torch
import yaml

from src.datamodule import dataset
from src.datamodule.data_factory import data_provider
from src.models import diffusion
from src.utils.parser import exp_parser
from src.utils.train import Trainer, get_expname_, setup_seed
from src.utils.schedule import get_schedule
import json


# root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings"
# root_pth = "/home/user/data/FrequencyDiffusion/savings"
setup_seed()


def prepare_train(model_config, data_config, args, n):
    root_pth = args["save_dir"]
    exp_name = (
        f"{get_expname_(model_config, data_config)}_{'t' if args['smoke_test'] else n}"
    )
    _, train_dl = data_provider(data_config, "train")
    _, val_dl = data_provider(data_config, "val")
    _, test_dl = data_provider(data_config, "test")

    # data_fn = getattr(dataset, data_config.pop("data_fn"))
    # train_dl, val_dl, test_dl, scaler = data_fn(data_config)

    df_ = model_config["diff_config"].pop("name")
    df = getattr(diffusion, df_)
    data_folder = os.path.join(
        root_pth, f"{args['data_config']}_{data_config['pred_len']}_{data_config['features']}"
    )
    save_folder = os.path.join(data_folder, df_, exp_name)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(model_config, w, indent=2)

    # with open(os.path.join(data_folder, "scaler.npy"), "wb") as f:
    #     np.save(f, scaler)
    torch.save(test_dl, os.path.join(data_folder, "test_dl.pt"))
    batch = next(iter(train_dl))
    seq_length, seq_channels = (
        batch["observed_data"].shape[1],
        batch["observed_data"].shape[2],
    )
    target_seq_length, target_seq_channels = (
        batch["future_data"].shape[1],
        batch["future_data"].shape[2],
    )
    ff = batch.get("future_features", None)
    future_seq_length, future_seq_channels = (
        ff.shape[1] if ff is not None else ff,
        ff.shape[2] if ff is not None else ff,
    )

    model_config["bb_config"]["seq_channels"] = target_seq_channels
    model_config["bb_config"]["seq_length"] = target_seq_length
    model_config["bb_config"]["latent_dim"] = model_config["cn_config"]["latent_dim"]
    model_config["cn_config"]["seq_channels"] = seq_channels
    model_config["cn_config"]["seq_length"] = seq_length
    model_config["cn_config"]["future_seq_channels"] = future_seq_channels
    model_config["cn_config"]["future_seq_length"] = future_seq_length
    ns_name = model_config["diff_config"].pop("noise_schedule")
    n_steps = (
        target_seq_length - 1 if df_ != "DDPM" else model_config["diff_config"]["T"]
    )
    noise_schedule = get_schedule(
        ns_name, args["data_config"], n_steps, train_dl, check_pth=data_folder
    )
    return model_config, noise_schedule, df, exp_name, save_folder, train_dl, val_dl


def main(args, n):
    device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)

    data_config = yaml.safe_load(
        open(f'configs/dataset/{args["data_config"]}.yaml', "r")
    )
    data_config = exp_parser(data_config, args)
    # data_fn = getattr(dataset, data_config.pop("data_fn"))

    model_config = yaml.safe_load(
        open(f'configs/model/{args["model_config"]}.yaml', "r")
    )
    model_config = exp_parser(model_config, args)

    model_config, noise_schedule, df, exp_name, save_folder, train_dl, val_dl = (
        prepare_train(model_config, data_config, args, n)
    )
    diff = df(
        backbone_config=model_config["bb_config"],
        conditioner_config=model_config["cn_config"],
        noise_schedule=noise_schedule,
        **model_config["diff_config"],
    )

    trainer = Trainer(
        smoke_test=args["smoke_test"],
        device=device,
        output_pth=save_folder,
        exp_name=exp_name,
        **model_config["train_config"],
    )

    trainer.fit(diff, train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument(
        "-mc",
        "--model_config",
        required=True,
        help="name of model configuration file.",
    )
    parser.add_argument(
        "-dc",
        "--data_config",
        type=str,
        required=True,
        help="name of data configuration file.",
    )
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)

    # Define overrides on models
    parser.add_argument("--diff_config.name", type=str)
    parser.add_argument("--diff_config.norm", action="store_true", default=None)
    parser.add_argument("--diff_config.pred_diff", action="store_true", default=None)
    parser.add_argument("--diff_config.noise_schedule", type=str)
    parser.add_argument("--bb_config.name", type=str)
    parser.add_argument("--bb_config.hidden_size", type=int)

    # Define overrides on dataset
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    args = parser.parse_args()
    for i in range(args.num_train):
        main(vars(args), i)
        if args.smoke_test:
            break
