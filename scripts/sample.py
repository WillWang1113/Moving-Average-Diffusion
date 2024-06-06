import argparse
import pandas as pd
import torch
import numpy as np
import os
import glob
from src.utils.filters import get_factors
from src.utils.train import Trainer, setup_seed
from src.utils.sample import plot_fcst, temporal_avg
from src.utils.metrics import calculate_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_seed()
print(device)


root_path = "/home/user/data/FrequencyDiffusion/savings/mfred"
# root_path = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings/mfred"
# dataset = "benchmarks"


def main(args):
    exp_path = os.path.join(root_path, args.model_name)

    if args.fast_sample:
        seq_length = 288
        factors = [i for i in range(2, seq_length + 1)]
        kernel_size = get_factors(seq_length) + [seq_length]
        sample_steps = [factors.index(i) for i in kernel_size]
        sample_steps.reverse()
    else:
        sample_steps = None
    kernel_size.sort()
    kernel_size.reverse()
    kernel_size += [1]
    print(kernel_size)
    # print(exp_path)

    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))

    # with open(os.path.join(root_path, "y_real.npy"), "rb") as f:
    #     y_real = np.load(f)

    with open(os.path.join(root_path, "scaler.npy"), "rb") as f:
        scaler = np.load(f, allow_pickle="TRUE").item()

    mean, std = scaler["data"]
    target = scaler["target"]

    df_out = []
    # exp_dirs = glob.glob(exp_path+"/*Backbone*")
    exp_dirs = glob.glob("*Backbone*", root_dir=exp_path)
    # exp_dirs.sort()
    exp_dirs = [e[:-2] for e in exp_dirs]
    exp_dirs = list(set(exp_dirs))
    exp_dirs.sort()
    print(exp_dirs)
    trainer = Trainer(smoke_test=args.smoke_test, device=device)

    for d in exp_dirs:
        # for d in exp_dirs:
        avg_m = []
        for i in range(args.num_train):
            # i = 't'
            read_d = os.path.join(exp_path, d + f"_{i}", "diffusion.pt")
            print(read_d)
            # print(kind)

            diff = torch.load(read_d)
            # diff = torch.compile(diff)
            # print(sum([p.numel() for p in diff.parameters()]))

            # DDIM on 50 steps
            # diff.fast_sample=True
            # diff.sigmas = torch.linspace(0, 0, diff.T, device=diff.device)
            # diff.sigmas = torch.linspace(1e-3, 1e-3, diff.T, device=diff.device)
            # print(diff.sigmas)
            # rest_steps = [x for x in range(diff.T) if x not in diff.sample_Ts]
            # sampled_numbers = random.sample(rest_steps, 20 - len(diff.sample_Ts))
            # sampled_numbers.extend(diff.sample_Ts)
            # sampled_numbers.sort(reverse=True)
            # diff.sample_Ts = sampled_numbers

            # FOR DIFFUSIONS
            if args.deterministic:
                sigmas = torch.zeros_like(diff.betas)
            else:
                # sigmas = None
                if d.__contains__("cold"):
                    print("COLD DiFFUSION can not introduce uncertainty")
                    sigmas = torch.zeros_like(diff.betas)
                else:
                    # sigmas = None
                    sigmas = torch.linspace(1e-3, 0.7, diff.T, device=diff.device)
                    # sigmas[0] = sigmas[1]
            diff.config_sampling(args.n_sample, sigmas, sample_steps, args.collect_all)
            y_pred, y_real = trainer.predict(diff, test_dl)
            print("finish predicting")
            assert len(y_real) == len(y_pred)
            # for j in range(len(y_pred)):
            #     y_pred[j] = y_pred[j].detach().cpu()
            #     y_real[j] = y_real[j].detach().cpu()

            y_pred = torch.concat(y_pred, dim=1)
            y_real = torch.concat(y_real, dim=0)

            y_pred = y_pred * std[target] + mean[target]
            y_real = y_real * std[target] + mean[target]

            # # PLOT WEIGHTS
            # cn = diff.backbone
            # for name, p in cn.named_parameters():
            #     print(name, p[0])
            #     break

            # if name.__contains__('weight'):
            #     fig, ax = plt.subplots()
            #     pos = ax.imshow(p.detach().cpu().numpy(), cmap='RdBu')
            #     cbar = fig.colorbar(pos, ax=ax)
            #     # cbar.minorticks_on()
            #     fig.tight_layout()
            #     fig.savefig('assets/'+name+'.png')

            if args.collect_all:
                y_pred, y_real = temporal_avg(y_pred, y_real, kernel_size, args.kind)

            m = calculate_metrics(y_pred, y_real)
            avg_m.append(m)
            if i in [0, "t"]:
                print("plotting")
                plot_fcst(
                    y_pred,
                    y_real,
                    kernel_size=kernel_size,
                    save_name=f"assets/{'fast_' if args.fast_sample else ''}{'dtm_' if args.deterministic else ''}{'multi_' if args.collect_all else ''}{d.split('/')[-1]}.png",
                )
            if args.smoke_test:
                break

        avg_m = np.array(avg_m)
        df = pd.DataFrame(
            avg_m.mean(axis=0, keepdims=False if args.collect_all else True),
            columns=["RMSE", "MAE", "CRPS", "MPBL"],
        )
        df["method"] = d.split("/")[-1]
        df["granularity"] = kernel_size if args.collect_all else 1
        df_out.append(df)

    df_out = pd.concat(df_out)
    df_out = df_out.reindex(
        columns=["method", "granularity", "RMSE", "MAE", "CRPS", "MPBL"]
    )
    # print(df_out)
    # df_out.to_csv("test.csv")
    df_out.to_csv(
        f"assets/{args.model_name}_{'fast' if args.fast_sample else ''}_{'dtm' if args.deterministic else ''}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--kind", type=str, required=True, choices=["freq", "time"])
    parser.add_argument("--num_train", type=int, default=5)
    # Define overrides with dot notation
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fast_sample", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--collect_all", action="store_true")
    parser.add_argument("--n_sample", type=int, default=50)
    args = parser.parse_args()
    main(args)
