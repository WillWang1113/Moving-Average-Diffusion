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
import src.benchmarks

# # torch.set_float32_matmul_precision('high')
# logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


# def main(args):
#     seed_everything(9)

#     # load configs
#     root_path = os.path.join(
#         args.save_dir, f"{args.dataset}_{args.pred_len}_{args.task}"
#     )
#     seq_length = args.pred_len
#     exp_path = os.path.join(root_path, args.model_name)
#     exp_config = os.path.join(exp_path, "config.json")
#     exp_config = json.load(open(exp_config, "rb"))
#     model_name = exp_config["diff_config"]["name"]
#     factor_only = exp_config["diff_config"]["factor_only"]
#     stride_equal_to_kernel_size = exp_config["diff_config"][
#         "stride_equal_to_kernel_size"
#     ]
#     model_class = getattr(models, model_name)
#     sample_steps = None

#     if factor_only:
#         kernel_size = get_factors(seq_length)
#         if args.start_ks is not None:
#             sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

#     else:
#         kernel_size = list(range(2, seq_length + 1))
#         # if args.start_ks is not None:
#         #     sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

#         if args.fast_sample:
#             factors = get_factors(seq_length)

#             # rest_ks = [x for x in factors if x not in kernel_size]
#             # extend_ks = random.sample(rest_ks, 50 - len(kernel_size))
#             # kernel_size.extend(extend_ks)
#             # kernel_size.sort()

#             sample_steps = [kernel_size.index(i) for i in factors]
#             if args.start_ks is not None:
#                 start_T = kernel_size.index(args.start_ks)
#                 if start_T in sample_steps:
#                     sample_steps = sample_steps[: sample_steps.index(start_T)]
#                 else:
#                     sample_steps += [start_T]
#             sample_steps.reverse()
#         else:
#             if args.start_ks is not None:
#                 sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

#     print("factor_only:\t", factor_only)
#     print("stride=ks:\t", stride_equal_to_kernel_size)
#     print("sample_steps:\t", sample_steps)

#     kernel_size.sort()
#     kernel_size.reverse()
#     kernel_size += [1]
#     # print(exp_path)

#     test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))

#     trainer = Trainer(
#         devices=[args.gpu], enable_progress_bar=False, fast_dev_run=args.smoke_test
#     )
#     avg_m = []

#     for i in range(args.num_train):
#         # # load low-res model
#         # if args.init_model is not None:
#         #     with open(f"checkpoints/{args.init_model}_{i}/args.txt", "r") as f:
#         #         model_args = json.load(f)
#         #     model_args = argparse.Namespace(**model_args)
#         #     init_model_class = getattr(src.benchmarks, model_args.model)

#         #     init_model = init_model_class.Model(configs=model_args)
#         #     init_model.load_state_dict(
#         #         torch.load(f"checkpoints/{args.init_model}_{i}/checkpoint.pth")
#         #     )
#         # else:
#         #     init_model = None

#         read_d = os.path.join(exp_path, f"best_model_path_{i}.pt")
#         best_model_path = torch.load(read_d)
#         diff = model_class.load_from_checkpoint(
#             best_model_path, map_location=f"cuda:{args.gpu}"
#         )

#         # FOR DIFFUSIONS
#         if args.deterministic:
#             sigmas = torch.zeros_like(diff.betas)
#         else:
#             sigmas = torch.linspace(0.2, 0.1, len(diff.betas))

#         # FOR MAD
#         diff.config_sampling(
#             args.n_sample,
#             w_cond=args.w_cond,
#             sigmas=sigmas,
#             sample_steps=sample_steps,
#             # init_model=init_model,
#         )

#         y_real = []
#         y_pred = trainer.predict(diff, test_dl)
#         y_pred = torch.concat(y_pred, dim=1)
#         for b in test_dl:
#             y_real.append(b["x"])
#         y_real = torch.concat(y_real)
#         if args.collect_all:
#             y_pred, y_real = temporal_avg(y_pred, y_real, kernel_size, args.kind)
#         else:
#             y_pred = y_pred.cpu().numpy()
#             y_real = y_real.cpu().numpy()
#         m, y_pred_q, y_pred_point = calculate_metrics(y_pred, y_real)
#         print(m)
#         avg_m.append(m)
#         out_name = [
#             # f"initmodel_{model_args.model}" if args.init_model else "",
#             f"startks{args.start_ks}" if args.start_ks else "",
#             "fast" if args.fast_sample else "",
#             "dtm" if args.deterministic else "",
#             f"{round(args.w_cond, 1)}" if args.model_name.__contains__("CFG") else "",
#             "multi" if args.collect_all else "",
#         ]
#         out_name = "_".join(out_name)
#         if i in [0, "t"]:
#             print("plotting")
#             plot_fcst(
#                 y_pred_q,
#                 y_real,
#                 y_pred_point=y_pred_point,
#                 kernel_size=kernel_size,
#                 save_name=os.path.join(
#                     exp_path,
#                     f"{out_name}.png",
#                 ),
#             )
#         if args.smoke_test:
#             break
#     avg_m = np.array(avg_m)
#     print(avg_m.mean(axis=0))
#     if args.num_train == 5:
#         np.save(
#             os.path.join(
#                 exp_path,
#                 f"{out_name}.npy",
#             ),
#             avg_m,
#         )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Hyperparameters config")
#     parser.add_argument(
#         "--save_dir", type=str, default="/home/user/data/FrequencyDiffusion/savings"
#     )
#     parser.add_argument("--dataset", type=str, default="etth2")
#     parser.add_argument("--pred_len", type=int, default=96)
#     parser.add_argument("--task", type=str, default="S", choices=["S", "M"])
#     parser.add_argument("--model_name", type=str, required=True)
#     # parser.add_argument("--kind", type=str, required=True, choices=["freq", "time"])

#     parser.add_argument("--gpu", type=int, default=0)
#     parser.add_argument("--num_train", type=int, default=5)
#     # Define overrides with dot notation
#     parser.add_argument("--deterministic", action="store_true")
#     parser.add_argument("--fast_sample", action="store_true")
#     parser.add_argument("--smoke_test", action="store_true")
#     parser.add_argument("--collect_all", action="store_true")
#     parser.add_argument("--n_sample", type=int, default=100)
#     parser.add_argument("--w_cond", type=float, default=1.0)
#     parser.add_argument("--start_ks", type=int, default=None)
#     # parser.add_argument("--init_model", type=str, default=None)
#     args = parser.parse_args()
#     main(args)


import argparse
import json
import os
import numpy as np
import torch
import yaml
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.datamodule.data_factory import data_provider
from src import models
from src.utils.filters import MovingAvgTime, get_factors
from src.utils.parser import exp_parser
from src.utils.schedule import get_schedule
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main(args):
    seed_everything(9, workers=True)

    data_config = yaml.safe_load(open(f"configs/dataset/{args.data_config}.yaml", "r"))
    data_config = exp_parser(data_config, vars(args))
    print(data_config)
    # data_config["condition"] = args.condition
    # data_config["kernel_size"] = args.kernel_size
    # data_config["seq_len"] = args.seq_len
    # data_config["pred_len"] = args.pred_len

    _, test_dl = data_provider(data_config, "test")

    # load configs
    root_path = os.path.join(
        args.save_dir, f"{args.data_config}_{data_config['pred_len']}_S"
    )
    exp_path = os.path.join(root_path, args.model_name)
    exp_config = os.path.join(exp_path, "config.json")
    exp_config = json.load(open(exp_config, "rb"))
    print(exp_config)
    model_name = exp_config["diff_config"]["name"]
    model_class = getattr(models, model_name)
    print(model_class)
    seq_length = data_config["pred_len"]
    factor_only = exp_config["diff_config"].get("factor_only", None)
    stride_equal_to_kernel_size = exp_config["diff_config"].get(
        "stride_equal_to_kernel_size", None
    )
    sample_steps = None

    if factor_only:
        args.fast_sample = False
        kernel_size = get_factors(seq_length)
        if args.start_ks is not None:
            sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

    else:
        kernel_size = list(range(2, seq_length + 1))

        if args.fast_sample:
            factors = get_factors(seq_length)

            sample_steps = [kernel_size.index(i) for i in factors]
            if args.start_ks is not None:
                start_T = kernel_size.index(args.start_ks)
                if start_T in sample_steps:
                    sample_steps = sample_steps[: sample_steps.index(start_T) + 1]
                else:
                    sample_steps += [start_T]
            sample_steps.reverse()
        else:
            if args.start_ks is not None:
                sample_steps = list(range(kernel_size.index(args.start_ks) + 1))

    # print(model_class.__name__)
    # return 0
    if model_class.__name__ == 'DDPM':
        sample_steps = None
        
    print(kernel_size)
    print("factor_only:\t", factor_only)
    print("stride=ks:\t", stride_equal_to_kernel_size)
    print("sample_steps:\t", sample_steps)
        
    trainer = Trainer(
        devices=[args.gpu], enable_progress_bar=False, fast_dev_run=args.smoke_test
    )
    avg_m = []

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
            w_cond=args.w_cond,
            sigmas=sigmas,
            sample_steps=sample_steps,
            condition=args.condition,
        )

        y_real = []
        y_pred = trainer.predict(diff, test_dl)
        y_pred = torch.concat(y_pred, dim=1)
        for b in test_dl:
            y_real.append(b["x"])
        y_real = torch.concat(y_real)
        y_pred = y_pred.cpu().numpy()
        y_real = y_real.cpu().numpy()
        m, y_pred_q, y_pred_point = calculate_metrics(y_pred, y_real)
        print(m)
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
