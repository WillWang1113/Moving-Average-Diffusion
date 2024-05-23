import json
import os
import random
import yaml
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.datamodule import dataset
from src.models import diffusion
from src.utils.train import get_expname_

root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings"
# root_pth = "/home/user/data/FrequencyDiffusion/savings"
fix_seed = 9
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config, run_args):
    device = f"cuda:{run_args['gpu']}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)

    data_fn = getattr(dataset, run_args["dataset"])
    train_dl, val_dl, test_dl, scaler = data_fn()
    if run_args["test"]:
        config["train_config"]["epochs"] = 5

    bb_, cn_, df_ = (
        config["bb_config"].get("name"),
        config["cn_config"].get("name"),
        config["diff_config"].get("name"),
    )
    df = getattr(diffusion, df_)

    exp_name = f"{get_expname_(config, bb_, cn_, df_)}"
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
        batch["conditions"]["observed_data"].shape[1],
        batch["conditions"]["observed_data"].shape[2],
    )
    target_seq_length, target_seq_channels = (
        batch["future_data"].shape[1],
        batch["future_data"].shape[2],
    )
    future_seq_length, future_seq_channels = (
        batch["conditions"]["future_features"].shape[1],
        batch["conditions"]["future_features"].shape[2],
    )

    print("\n")
    print("MODEL PARAM:")
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

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["train_config"]["epochs"],
        check_val_every_n_epoch=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            ),
            checkpoint_callback,
        ],
        default_root_dir=save_folder,
        # fast_dev_run=True
    )
    trainer.fit(
        diff,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )
    # diff.configure_sampling()
    # preds = trainer.predict(diff, test_dl)


if __name__ == "__main__":
    # FOR MY OWN EXPERIMENTS
    args = {"dataset": "mfred", "num_train": 1, "test": True, "gpu": 0}

    # for beta in [0, 1]:
    #     for bb_name in ["MLPBackbone", "ResNetBackbone"]:
    #         for freq in [True, False]:
    #             for diff in [True, False]:
    #                 n = 1 if args["test"] else args["num_train"]
    #                 for i in range(n):
    #                     config = yaml.safe_load(open("configs/default.yaml", "r"))
    #                     config["bb_config"]["name"] = bb_name
    #                     if bb_name == "ResNetBackbone":
    #                         config["bb_config"]["hidden_size"] = 128
    #                     config["diff_config"]["noise_kw"]["min_beta"] = beta
    #                     config["diff_config"]["noise_kw"]["max_beta"] = beta
    #                     config["diff_config"]["freq_kw"]["frequency"] = freq
    #                     config["diff_config"]["freq_kw"]["real_imag"] = freq
    #                     config["diff_config"]["fit_on_diff"] = diff
    #                     main(config, args, i)

    config = yaml.safe_load(open("configs/default.yaml", "r"))
    main(config, args)
    # config = yaml.safe_load(open("configs/default.yaml", "r"))
    # main(config, args, 1)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset", type=str, default="mfred")
    # parser.add_argument("-n", "--num_train", type=int, default=5)
    # parser.add_argument("-s", "--setting", type=str, default="default_mfred")
    # parser.add_argument("-t", "--test", action="store_true")
    # parser.add_argument("--gpu", type=int, default=0)
    # args = parser.parse_args()
