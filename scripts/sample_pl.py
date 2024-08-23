import argparse
import json
import pandas as pd
import torch
import numpy as np
import os
import logging
from tqdm import tqdm
from gluonts.torch.model.predictor import PyTorchPredictor
from src.utils.filters import get_factors
from src.models.diffusion_pl import MADFreq, MADTime
from src.benchmarks.PatchTST import Model
from lightning import Trainer
from lightning.fabric import seed_everything
from src.utils.sample import plot_fcst, temporal_avg
from src.utils.metrics import calculate_metrics

# torch.set_float32_matmul_precision('high')
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


def main(args):
    # quantiles = [0.05 * (1 + i) for i in range(19)]
    root_path = os.path.join(
        args.save_dir, f"{args.dataset}_{args.pred_len}_{args.task}"
    )
    # device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # print("Using: ", device)
    seed_everything(9)
    seq_length = args.pred_len
    exp_path = os.path.join(root_path, args.model_name)

    if args.fast_sample:
        factors = [i for i in range(2, seq_length + 1)]
        kernel_size = get_factors(seq_length) + [seq_length]

        # rest_ks = [x for x in factors if x not in kernel_size]
        # extend_ks = random.sample(rest_ks, 50 - len(kernel_size))
        # kernel_size.extend(extend_ks)
        # kernel_size.sort()

        sample_steps = [factors.index(i) for i in kernel_size]
        sample_steps.reverse()
    else:
        kernel_size = [i for i in range(2, seq_length + 1)]
        sample_steps = None
    kernel_size.sort()
    kernel_size.reverse()
    kernel_size += [1]
    # print(exp_path)

    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))

    trainer = Trainer(devices=[args.gpu], enable_progress_bar=False, fast_dev_run=args.smoke_test)
    avg_m = []

    for i in range(args.num_train):
        # /home/user/data/FrequencyDiffusion/savings/etth1_96_S/MADfreq_v2/lightning_logs/version_0/checkpoints/epoch=8-step=603.ckpt

        read_d = os.path.join(exp_path, f"best_model_path_{i}.pt")

        # model_parser = argparse.ArgumentParser()
        # model_args = model_parser.parse_args()
        with open("base_ckpt/args.txt", "r") as f:
            model_args = json.load(f)
        model_args = argparse.Namespace(**model_args)
        
        if args.init_model:
            init_model = Model(configs=model_args)
            init_model.load_state_dict(torch.load("base_ckpt/checkpoint.pth"))
        else:
            init_model = None

        best_model_path = torch.load(read_d)
        if args.kind == "freq":
            diff = MADFreq.load_from_checkpoint(best_model_path)
        else:
            diff = MADTime.load_from_checkpoint(best_model_path)
        # diff = diff.load

        # diff = torch.load(read_d)
        # FOR DIFFUSIONS
        if args.deterministic:
            sigmas = torch.zeros_like(diff.betas)
        else:
            sigmas = torch.linspace(0.2, 0.1, len(diff.betas))

        diff.config_sampling(
            args.n_sample, sigmas, sample_steps, args.collect_all, init_model=init_model
        )
        y_real = []
        y_pred = trainer.predict(diff, test_dl)
        y_pred = torch.concat(y_pred, dim=1)
        # for _ in tqdm(range(args.n_sample)):
        #     y_p = trainer.predict(diff, test_dl)
        #     y_p = torch.concat(y_p)
        #     y_pred.append(y_p)
        # y_pred = torch.stack(y_pred)
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
        if i in [0, "t"]:
            print("plotting")
            plot_fcst(
                y_pred_q,
                y_real,
                y_pred_point=y_pred_point,
                kernel_size=kernel_size,
                save_name=os.path.join(
                    exp_path,
                    f"{'fast_' if args.fast_sample else ''}{'dtm_' if args.deterministic else ''}{'multi_' if args.collect_all else ''}.png",
                ),
                # save_name=f"assets/{args.dataset}_{args.pred_len}_{args.model_name}_{'fast_' if args.fast_sample else ''}{'dtm_' if args.deterministic else ''}{'multi_' if args.collect_all else ''}{d.split('/')[-1]}.png",
            )
        if args.smoke_test:
            break
    avg_m = np.array(avg_m)
    df = pd.DataFrame(
        avg_m.mean(axis=0, keepdims=False if args.collect_all else True),
        columns=["MAE", "MSE", "CRPS"],
        # columns=["RMSE", "MAE", "CRPS", "MPBL"],
    )
    print(df)
    # df["method"] = d.split("/")[-1]
    df["granularity"] = kernel_size if args.collect_all else 1
    df.to_csv(
        os.path.join(
            exp_path,
            f"{'fast_' if args.fast_sample else ''}{'dtm_' if args.deterministic else ''}{'multi_' if args.collect_all else ''}.csv",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--pred_len", required=True, type=int)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--kind", type=str, required=True, choices=["freq", "time"])

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)
    # Define overrides with dot notation
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fast_sample", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--collect_all", action="store_true")
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--init_model", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
