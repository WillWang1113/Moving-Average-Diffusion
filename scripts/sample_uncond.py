import argparse
import json
import os

import numpy as np
import torch
import yaml

# from src.models.diffusion_pl import MADFreq, MADTime
from lightning import Trainer
from lightning.fabric import seed_everything

from src import models
from src.datamodule.data_factory import data_provider
from src.utils.filters import MovingAvgTime, get_factors
from src.utils.metrics import ablation_metrics, get_encoder
from src.utils.parser import exp_parser

# logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)


def main(args):
    device = f"cuda:{args.gpu}"
    seed_everything(9, workers=True)

    data_config = yaml.safe_load(open(f"configs/dataset/{args.data_config}.yaml", "r"))
    data_config = exp_parser(data_config, vars(args))
    data_config["features"] = "S"
    data_config["batch_size"] = 512
    data_config["condition"] = None
    # data_config["pred_len"] = 576

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
        # f"startks_{args.start_ks}",
        # f"fast_{args.fast_sample}",
        f"dtm_{args.deterministic}",
        # ! Force align
        # f"rds_{args.num_diff_steps}",
        # f"{round(args.w_cond, 1)}" if args.model_name.__contains__("CFG") else "",
    ]
    out_name = "_".join(out_name)
    print(out_name)

    _, train_dl = data_provider(data_config, "train")
    _, test_dl = data_provider(data_config, "test")
    ae_ckpt = os.path.join(root_path, "ae.ckpt")
    if os.path.exists(ae_ckpt):
        print("Load pretrained AutoEncoder")
        ae = torch.load(ae_ckpt)
    else:
        ae = get_encoder(train_dl, device)
        torch.save(ae, ae_ckpt)
    print("Load data")

    sample_steps = None
    if args.fast_sample:
        noise_schedule = torch.load(
            os.path.join(
                root_path,
                f"std_sched_{diffusion_steps}.pt",
            )
        )
        all_K = noise_schedule["alphas"]
        all_factors = get_factors(seq_length)
        sample_steps = []
        for i in all_factors:
            ideal_K = MovingAvgTime(i, seq_length).K.unsqueeze(0)
            close_id = torch.argmin(
                ((all_K.flatten(1) - ideal_K.flatten(1)) ** 2).mean(dim=1)
            ).int()
            sample_steps.append(close_id.item())
        if 0 not in sample_steps:
            sample_steps.append(0)
    print(sample_steps)

    trainer = Trainer(devices=[args.gpu], fast_dev_run=args.smoke_test)
    avg_m = []

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
        # diff.to(device)
        y_real = []
        for batch in test_dl:
            y_real.append(batch["x"].cpu())


        y_syn = trainer.predict(diff, test_dl)
        y_syn = torch.concat(y_syn, dim=1).detach()
        # y_pred = trainer.predict(diff, test_dl)

        if args.data_config == "solar":
            y_syn = torch.where(y_syn < -6.58696556e-01, -6.58696556e-01, y_syn)
        # print(y_syn.shape)
        y_real = torch.concat(y_real).detach()

        all_m = []
        for n in range(args.n_sample):
            if args.model_name.__contains__("DDPM"):
                y_syn[n] = (y_syn[n] - y_syn[n].mean(dim=1, keepdim=True)) / torch.sqrt(
                    torch.var(y_syn[n], dim=1, keepdim=True, unbiased=False) + 1e-6
                )
                y_syn[n] = (
                    y_syn[n]
                    * torch.sqrt(
                        torch.var(y_real, dim=1, keepdim=True, unbiased=False) + 1e-6
                    )
                ) + y_real.mean(dim=1, keepdim=True)

            m_stat = ablation_metrics(y_syn[n], y_real, ae)
            # m_stat = context_fid(y_syn[n], y_real, ae)
            all_m.append(np.array(m_stat).reshape(1, -1))
        all_m = np.concatenate(all_m).mean(axis=0)
        print(all_m)
        # if i == 0:
        try:
            np.save(
                os.path.join(
                    exp_path,
                    f"{out_name}_syn_{i}.npy",
                ),
                y_syn.squeeze(0).cpu().numpy(),
            )
        except:
            print("Cannot save samples")

    try:
        np.save(
            os.path.join(
                exp_path,
                f"{out_name}_real.npy",
            ),
            y_real.squeeze(0).cpu().numpy(),
        )
    except:
        print("Cannot save")


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
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--w_cond", type=float, default=0.5)
    parser.add_argument("--fast_sample", action="store_true")

    # parser.add_argument("--num_diff_steps", type=float, default=1.0)

    # Define overrides on dataset
    parser.add_argument("--condition", type=str, choices=["sr", "fcst"])
    # parser.add_argument("--start_ks", type=int)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_train", type=int, default=5)
    args = parser.parse_args()
    main(args)
