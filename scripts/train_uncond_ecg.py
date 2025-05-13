import argparse
import json
import os
import numpy as np
import torch
import torch.utils
import torch.utils.data
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

    all_data = np.loadtxt(
        os.path.join(data_config["root_path"], data_config["data_path"]), delimiter=","
    )
    all_data = torch.from_numpy(all_data)
    all_data = torch.nn.functional.interpolate(all_data.unsqueeze(1), size=24)

    all_data = all_data.permute(0, 2, 1).float()

    num_train = int(0.7 * len(all_data))
    num_test = int(0.1 * len(all_data))
    num_val = len(all_data) - num_train - num_test
    train_data = all_data[:num_train]
    val_data = all_data[num_train : num_train + num_val]
    test_data = all_data[-num_test:]

    class TempDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __getitem__(self, index):
            return {"x": self.data[index]}

        def __len__(self):
            return len(self.data)

    train_ds = TempDataset(train_data)
    val_ds = TempDataset(val_data)
    # test_ds = TempDataset(test_data)
    train_dl = torch.utils.data.DataLoader(
        train_ds, shuffle=True, batch_size=data_config["batch_size"]
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, shuffle=True, batch_size=data_config["batch_size"]
    )


    data_folder = os.path.join(
        root_pth,
        f"{args['data_config']}_24_{data_config['features']}",
    )
    save_folder = os.path.join(data_folder, args["model_config"])
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(model_config, w, indent=2)

    df_ = model_config["diff_config"].pop("name")
    df = getattr(models, df_)
    # print(df)

    batch = next(iter(train_dl))
    target_seq_length, target_seq_channels = (batch["x"].shape[1], batch["x"].shape[2])
    model_config["bb_config"]["seq_channels"] = target_seq_channels
    model_config["bb_config"]["seq_length"] = target_seq_length

    if data_config["condition"] is not None:
        seq_length, seq_channels = (batch["c"].shape[1], batch["c"].shape[2])
        model_config["bb_config"]["cond_seq_chnl"] = seq_channels
        model_config["bb_config"]["cond_seq_len"] = seq_length

    ns_name = model_config["diff_config"].pop("noise_schedule")
    n_steps = model_config["diff_config"]["T"]

    ns_path = get_schedule(
        ns_name,
        n_steps,
        check_pth=data_folder,
        train_dl=train_dl,
    )
    return model_config, ns_path, df, save_folder, train_dl, val_dl


def main(args, n):
    # MUST SETUP SEED AFTER prepare_train
    seed_everything(n, workers=True)

    data_config = yaml.safe_load(
        open(f"configs/dataset/{args['data_config']}.yaml", "r")
    )
    data_config = exp_parser(data_config, args)

    model_config = yaml.safe_load(
        open(f"configs/model/{args['model_config']}.yaml", "r")
    )
    model_config = exp_parser(model_config, args)

    model_config, ns_path, df, save_folder, train_dl, val_dl = prepare_train(
        model_config, data_config, args, n
    )
    print(ns_path)
    diff = df(
        backbone_config=model_config["bb_config"],
        # conditioner_config=model_config["cn_config"],
        ns_path=ns_path,
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
        check_val_every_n_epoch=model_config["train_config"]["val_step"],
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
    parser.add_argument("--save_dir", default="/home/user/data/MAD/savings", type=str)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)

    # override data params
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    # parser.add_argument("--condition", type=str)
    # parser.add_argument("--kernel_size", type=int)

    args = parser.parse_args()
    for i in range(args.num_train):
        main(vars(args), i)
        if args.smoke_test:
            break
