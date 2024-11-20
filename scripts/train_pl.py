import argparse
import json
import os

import torch
import yaml
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.datamodule.data_factory import data_provider
from src import models
from src.utils.parser import exp_parser
from src.utils.schedule import get_schedule


def prepare_train(model_config, data_config, args, n):
    root_pth = args["save_dir"]

    _, train_dl = data_provider(data_config, "train")
    _, val_dl = data_provider(data_config, "val")
    _, test_dl = data_provider(data_config, "test")

    data_folder = os.path.join(
        root_pth,
        f"{args['data_config']}_{data_config['pred_len']}_{data_config['features']}",
    )
    save_folder = os.path.join(
        data_folder, args["model_config"] + f"_bs{data_config['batch_size']}"
    )
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(model_config, w, indent=2)

    df_ = model_config["diff_config"].pop("name")
    df = getattr(models, df_)
    print(df)
    # save_folder = os.path.join(data_folder, df_, exp_name)

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
    model_config["cn_config"]["target_seq_length"] = target_seq_length
    model_config["cn_config"]["target_seq_channels"] = target_seq_channels
    ns_name = model_config["diff_config"].pop("noise_schedule")
    n_steps = (
        target_seq_length - 1
        if df_.__contains__("MAD")
        else model_config["diff_config"]["T"]
    )
    noise_schedule = get_schedule(
        ns_name,
        n_steps,
        data_name=args["data_config"],
        train_dl=train_dl,
        check_pth=data_folder,
        seq_len=target_seq_length,
        factor_only=model_config["diff_config"]["factor_only"],
        stride_equal_to_kernel_size=model_config["diff_config"][
            "stride_equal_to_kernel_size"
        ],
    )
    return model_config, noise_schedule, df, save_folder, train_dl, val_dl, test_dl


def main(args, n):
    data_config = yaml.safe_load(
        open(f'configs/dataset/{args["data_config"]}.yaml', "r")
    )
    data_config = exp_parser(data_config, args)

    model_config = yaml.safe_load(
        open(f'configs/model/{args["model_config"]}.yaml', "r")
    )
    model_config = exp_parser(model_config, args)

    model_config, noise_schedule, df, save_folder, train_dl, val_dl, test_dl = (
        prepare_train(model_config, data_config, args, n)
    )
    # MUST SETUP SEED AFTER prepare_train
    seed_everything(n, workers=True)
    diff = df(
        backbone_config=model_config["bb_config"],
        conditioner_config=model_config["cn_config"],
        noise_schedule=noise_schedule,
        lr=model_config["train_config"]["lr"],
        alpha=model_config["train_config"]["alpha"],
        **model_config["diff_config"],
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=model_config["train_config"]["early_stop"],
    )
    mc = ModelCheckpoint(monitor="val_loss", dirpath=save_folder, save_top_k=1)
    trainer = Trainer(
        max_epochs=model_config["train_config"]["epochs"],
        deterministic=True,
        devices=[args["gpu"]],
        callbacks=[es, mc],
        default_root_dir=save_folder,
        fast_dev_run=args["smoke_test"],
        enable_progress_bar=args["smoke_test"],
        check_val_every_n_epoch=model_config["train_config"]['val_step']
        # **model_config["train_config"],
    )

    trainer.fit(diff, train_dl, val_dl)

    torch.save(mc.best_model_path, os.path.join(save_folder, f"best_model_path_{n}.pt"))

    print(mc.best_model_path)


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

    # Define overrides on dataset
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    for i in range(args.num_train):
        main(vars(args), i)
        if args.smoke_test:
            break
