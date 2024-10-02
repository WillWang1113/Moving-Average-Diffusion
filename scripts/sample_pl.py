import argparse
import json
import pandas as pd
import torch
import numpy as np
import os
import logging
from src.utils.filters import get_factors
from src import models
from src.benchmarks.PatchTST import Model

# from src.models.diffusion_pl import MADFreq, MADTime
from lightning import Trainer
from lightning.fabric import seed_everything
from src.utils.sample import plot_fcst, temporal_avg
from src.utils.metrics import calculate_metrics

# torch.set_float32_matmul_precision('high')
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


def main(args):
    seed_everything(9)

    # load configs
    root_path = os.path.join(
        args.save_dir, f"{args.dataset}_{args.pred_len}_{args.task}"
    )
    seq_length = args.pred_len
    exp_path = os.path.join(root_path, args.model_name)
    exp_config = os.path.join(exp_path, "config.json")
    exp_config = json.load(open(exp_config, "rb"))
    model_name = exp_config["diff_config"]["name"]
    factor_only = exp_config["diff_config"]["factor_only"]
    stride_equal_to_kernel_size = exp_config["diff_config"][
        "stride_equal_to_kernel_size"
    ]
    model_class = getattr(models, model_name)
    sample_steps = None

    if factor_only:
        kernel_size = get_factors(seq_length)
        if args.start_ks is not None:
            sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

    else:
        kernel_size = list(range(2, seq_length + 1))
        # if args.start_ks is not None:
        #     sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

        if args.fast_sample:
            factors = get_factors(seq_length)

            # rest_ks = [x for x in factors if x not in kernel_size]
            # extend_ks = random.sample(rest_ks, 50 - len(kernel_size))
            # kernel_size.extend(extend_ks)
            # kernel_size.sort()

            sample_steps = [kernel_size.index(i) for i in factors]
            if args.start_ks is not None:
                start_T = kernel_size.index(args.start_ks)
                if start_T in sample_steps:
                    sample_steps = sample_steps[: sample_steps.index(start_T)]
                else:
                    sample_steps += [start_T]
            sample_steps.reverse()
        else:
            if args.start_ks is not None:
                sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

    print("factor_only:\t", factor_only)
    print("stride=ks:\t", stride_equal_to_kernel_size)
    print("sample_steps:\t", sample_steps)

    kernel_size.sort()
    kernel_size.reverse()
    kernel_size += [1]
    # print(exp_path)

    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))

    trainer = Trainer(
        devices=[args.gpu], enable_progress_bar=False, fast_dev_run=args.smoke_test
    )
    avg_m = []

    if args.init_model is not None:
        with open(f"checkpoints/{args.init_model}/args.txt", "r") as f:
            model_args = json.load(f)
        model_args = argparse.Namespace(**model_args)
        init_model = Model(configs=model_args)
        init_model.load_state_dict(
            torch.load(f"checkpoints/{args.init_model}//checkpoint.pth")
        )
    else:
        init_model = None

    for i in range(args.num_train):
        read_d = os.path.join(exp_path, f"best_model_path_{i}.pt")

        best_model_path = torch.load(read_d)
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
            sigmas=sigmas,
            sample_steps=sample_steps,
            init_model=init_model,
        )

        y_real = []
        y_pred = trainer.predict(diff, test_dl)
        y_pred = torch.concat(y_pred, dim=1)
        for b in test_dl:
            y_real.append(b["future_data"])
        y_real = torch.concat(y_real)
        if args.collect_all:
            y_pred, y_real = temporal_avg(y_pred, y_real, kernel_size, args.kind)
        else:
            y_pred = y_pred.cpu().numpy()
            y_real = y_real.cpu().numpy()
        m, y_pred_q, y_pred_point = calculate_metrics(y_pred, y_real)
        print(m)
        avg_m.append(m)
        out_name = [
            f"initmodel_startks{args.start_ks}" if args.init_model else "",
            "fast" if args.fast_sample else "",
            "dtm" if args.deterministic else "",
            "multi" if args.collect_all else "",
        ]
        out_name = "_".join(out_name)
        if i in [0, "t"]:
            print("plotting")
            plot_fcst(
                y_pred_q,
                y_real,
                y_pred_point=y_pred_point,
                kernel_size=kernel_size,
                save_name=os.path.join(
                    exp_path,
                    f"{out_name}.png",
                ),
            )
        if args.smoke_test:
            break
    avg_m = np.array(avg_m)
    print(avg_m.mean(axis=0))
    if args.num_train == 5:
        np.save(
            os.path.join(
                exp_path,
                f"{out_name}.npy",
            ),
            avg_m,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument(
        "--save_dir", type=str, default="/home/user/data/FrequencyDiffusion/savings"
    )
    parser.add_argument("--dataset", type=str, default="etth2")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--task", type=str, default="S", choices=["S", "M"])
    parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--kind", type=str, required=True, choices=["freq", "time"])

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)
    # Define overrides with dot notation
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fast_sample", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--collect_all", action="store_true")
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--start_ks", type=int, default=None)
    parser.add_argument("--init_model", type=str, default=None)
    args = parser.parse_args()
    main(args)
