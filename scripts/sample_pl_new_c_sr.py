import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
import yaml

# from src.models.diffusion_pl import MADFreq, MADTime
from lightning import Trainer
from lightning.fabric import seed_everything

from src import models
from src.datamodule.data_factory import data_provider
from src.utils.filters import get_factors
from src.utils.metrics import calculate_metrics, sr_metrics
from src.utils.parser import exp_parser
from src.utils.sample import plot_fcst
import logging

# logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)


def main(args):
    seed_everything(9, workers=True)

    data_config = yaml.safe_load(open(f"configs/dataset/{args.data_config}.yaml", "r"))
    data_config = exp_parser(data_config, vars(args))
    print(data_config)

    # load configs
    root_path = os.path.join(
        args.save_dir, f"{args.data_config}_{data_config['pred_len']}_S"
    )
    exp_path = os.path.join(root_path, args.model_name)
    exp_config = os.path.join(exp_path, "config.json")
    exp_config = json.load(open(exp_config, "rb"))
    model_name = exp_config["diff_config"]["name"]
    model_class = getattr(models, model_name)
    seq_length = data_config["pred_len"]
    diffusion_steps = exp_config["diff_config"]["T"]
    out_name = [
        # f"initmodel_{model_args.model}" if args.init_model else "",
        f"cond_{args.condition}",
        f"startks_{args.start_ks}",
        f"fast_{args.fast_sample}",
        f"dtm_{args.deterministic}",
        # f"{round(args.w_cond, 1)}" if args.model_name.__contains__("CFG") else "",
    ]
    out_name = "_".join(out_name)
    if (args.num_train == 5) and os.path.exists(
        os.path.join(
            exp_path,
            f"{out_name}.npy",
        ),
    ):
        print("already have metrics.")
        # return 0
    _, test_dl = data_provider(data_config, "test")
    print("Load data")
    sample_steps = list(range(diffusion_steps))
    sample_steps.reverse()

    kernel_size = get_factors(seq_length)
    # kernel_size = np.array(kernel_size)
    interp_k = int((diffusion_steps - len(kernel_size)) / (len(kernel_size) - 1))

    if args.fast_sample:
        sample_steps = sample_steps[:: interp_k + 1]
        if 0 not in sample_steps:
            sample_steps.append(0)

    if args.start_ks is not None:
        start_idx = kernel_size.index(args.start_ks)
        start_idx = round(start_idx / (len(kernel_size) - 1) * (len(sample_steps) - 1))
        start_idx = len(sample_steps) - 1 - start_idx
        sample_steps = sample_steps[start_idx:]

    if model_class.__name__ == "DDPM":
        sample_steps = None

    print(kernel_size)
    # print("factor_only:\t", factor_only)
    # print("stride=ks:\t", stride_equal_to_kernel_size)
    print("sample_steps:\t", sample_steps)

    trainer = Trainer(devices=[args.gpu], fast_dev_run=args.smoke_test)
    avg_m = []
    device = f"cuda:{args.gpu}"

    for i in range(args.num_train):
        print("runtime:\t", i)
        read_d = os.path.join(exp_path, f"best_model_path_{i}.pt")
        best_model_path = torch.load(read_d)
        best_model_path = os.path.join(exp_path, best_model_path.split("/")[-1])
        diff = model_class.load_from_checkpoint(
            best_model_path, map_location=f"cuda:{args.gpu}"
        )

        # FOR DIFFUSIONS
        if args.deterministic:
            sigmas = torch.zeros_like(diff.betas)
        else:
            sigmas = torch.linspace(0.2, 0.1, len(diff.betas))

        # FOR MAD
        diff.config_sampling(
            args.n_sample,
            w_cond=args.w_cond,
            sigmas=sigmas,
            sample_steps=sample_steps,
            condition=args.condition,
        )
        diff.eval()
        print("Finish Config Model")

        y_real = []
        # y_pred_all = []
        for batch in test_dl:
            #     for k in batch:
            #         batch[k] = batch[k].to(f"cuda:{args.gpu}")

            y_real.append(batch["x"].cpu())
            if args.smoke_test:
                break
        #     y_pred = diff.predict_step(batch, 0)
        #     y_pred_all.append(y_pred.cpu())

        y_pred = trainer.predict(diff, test_dl)
        y_pred = torch.concat(y_pred, dim=1).detach()
        if args.n_sample > 1:
            y_pred = torch.quantile(y_pred, q=0.5, dim=0, keepdim=True)
        # if i == 0:

        y_real = torch.concat(y_real).detach()
        print(y_pred.shape)
        print(y_real.shape)
        print("Calculate Metrics")
        m = sr_metrics(y_pred, y_real, args.kernel_size)
        avg_m.append(m)
        out_name = [
            # f"initmodel_{model_args.model}" if args.init_model else "",
            f"cond_{args.condition}",
            f"startks_{args.start_ks}",
            f"fast_{args.fast_sample}",
            f"dtm_{args.deterministic}",
            # f"{round(args.w_cond, 1)}" if args.model_name.__contains__("CFG") else "",
        ]
        out_name = "_".join(out_name)

        if i in [0, "t"]:
            print("plotting")
            plot_fcst(
                y_pred.squeeze(0).cpu().numpy(),
                y_real.cpu().numpy(),
                y_pred_point=y_pred,
                kernel_size=kernel_size,
                save_name=os.path.join(
                    exp_path,
                    f"{out_name}.png",
                ),
            )
            try:
                np.save(
                    os.path.join(
                        exp_path,
                        f"{out_name}_pred.npy",
                    ),
                    y_pred.squeeze(0).cpu().numpy(),
                )
            except:
                print('Cannot save')
        if args.smoke_test:
            break
    avg_m = np.array(avg_m)
    print(avg_m.mean(axis=0))
    if (args.num_train == 5) and (not args.smoke_test):
        np.save(
            os.path.join(
                exp_path,
                f"{out_name}.npy",
            ),
            avg_m,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument("--save_dir", type=str, default="/home/user/data/MAD/savings")
    parser.add_argument(
        "-dc",
        "--data_config",
        type=str,
        required=True,
        help="name of data configuration file.",
    )
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--gpu", type=int, default=0)
    # Define overrides with dot notation
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fast_sample", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--w_cond", type=float, default=0.5)

    # Define overrides on dataset
    parser.add_argument("--condition", type=str, choices=["sr", "fcst"])
    parser.add_argument("--start_ks", type=int)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_train", type=int, default=5)
    args = parser.parse_args()
    main(args)
