import argparse
from datetime import datetime
import random
import pandas as pd
import torch
import numpy as np
import os
import glob
from src.utils.filters import get_factors
from src.utils.train import Trainer, setup_seed
from src.utils.sample import plot_fcst, temporal_avg
from src.utils.metrics import calculate_metrics

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)


# root_path = "/home/user/data/FrequencyDiffusion/savings/mfred"
# root_path = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings/mfred"
# dataset = "benchmarks"


def main(args):
    quantiles = [0.05 * (1 + i) for i in range(19)]
    root_path = os.path.join(
        args.save_dir, f"{args.dataset}_{args.pred_len}_{args.task}"
    )
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)
    setup_seed()
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

    # df_out = []
    # exp_dirs = glob.glob(exp_path+"/diffusion_*.pt")
    # exp_dirs = glob.glob(exp_path+"/*Backbone*")
    # exp_dirs = glob.glob("*Backbone*fcst*", root_dir=exp_path)
    # exp_dirs.sort()
    # exp_dirs = [e[:-2] for e in exp_dirs]
    # exp_dirs = list(set(exp_dirs))
    # exp_dirs.sort()
    # print(exp_dirs)
    # return 0
    trainer = Trainer(smoke_test=args.smoke_test, device=device)
    avg_m = []
    for i in range(args.num_train):
        read_d = os.path.join(exp_path, f"diffusion_{i}.pt")
        diff = torch.load(read_d)
        # FOR DIFFUSIONS
        if args.deterministic:
            sigmas = torch.zeros_like(diff.betas)
        else:
            sigmas = torch.linspace(0.2, 0.1, len(diff.betas))

        diff.config_sampling(args.n_sample, sigmas, sample_steps, args.collect_all)
        y_pred, y_real = trainer.predict(diff, test_dl)
        print("finish predicting")
        assert len(y_real) == len(y_pred)
        y_pred = torch.concat(y_pred, dim=1)
        y_real = torch.concat(y_real, dim=0)
        if args.collect_all:
            y_pred, y_real = temporal_avg(y_pred, y_real, kernel_size, args.kind)
        else:
            y_pred = y_pred.cpu().numpy()
            y_real = y_real.cpu().numpy()
        m = calculate_metrics(y_pred, y_real, quantiles)
        print(m)
        avg_m.append(m)
        if i in [0, "t"]:
            print("plotting")
            plot_fcst(
                y_pred,
                y_real,
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
    # df["method"] = d.split("/")[-1]
    df["granularity"] = kernel_size if args.collect_all else 1
    df.to_csv(
        os.path.join(
            exp_path,
            f"{'fast_' if args.fast_sample else ''}{'dtm_' if args.deterministic else ''}{'multi_' if args.collect_all else ''}.csv",
        )
    )

    # for d in exp_dirs:
    #     # for d in exp_dirs:
    #     avg_m = []
    #     for i in range(args.num_train):
    #         read_d = os.path.join(exp_path, d + f"_{i}", "diffusion.pt")
    #         print(read_d)
    #         # print(kind)

    #         diff = torch.load(read_d)
    #         # diff = torch.compile(diff)
    #         # print(sum([p.numel() for p in diff.parameters()]))

    #         # FOR DIFFUSIONS
    #         if args.deterministic:
    #             sigmas = torch.zeros_like(diff.betas)
    #         else:
    #             # sigmas = None
    #             if d.__contains__("cold"):
    #                 print("COLD DiFFUSION can not introduce uncertainty")
    #                 sigmas = torch.zeros_like(diff.betas)
    #             else:
    #                 # sigmas = None
    #                 sigmas = torch.linspace(0.2, 0.1, len(diff.betas))
    #                 # sigmas = torch.linspace(1e-3, 0.7, diff.T, device=diff.device)
    #                 # sigmas[0] = sigmas[1]
    #         diff.config_sampling(args.n_sample, sigmas, sample_steps, args.collect_all)
    #         y_pred, y_real = trainer.predict(diff, test_dl)
    #         print("finish predicting")
    #         assert len(y_real) == len(y_pred)
    #         # for j in range(len(y_pred)):
    #         #     y_pred[j] = y_pred[j].detach().cpu()
    #         #     y_real[j] = y_real[j].detach().cpu()

    #         y_pred = torch.concat(y_pred, dim=1)
    #         y_real = torch.concat(y_real, dim=0)

    #         # y_pred = y_pred * std[target] + mean[target]
    #         # y_real = y_real * std[target] + mean[target]

    #         # # PLOT WEIGHTS
    #         # cn = diff.backbone
    #         # for name, p in cn.named_parameters():
    #         #     print(name, p[0])
    #         #     break

    #         # if name.__contains__('weight'):
    #         #     fig, ax = plt.subplots()
    #         #     pos = ax.imshow(p.detach().cpu().numpy(), cmap='RdBu')
    #         #     cbar = fig.colorbar(pos, ax=ax)
    #         #     # cbar.minorticks_on()
    #         #     fig.tight_layout()
    #         #     fig.savefig('assets/'+name+'.png')

    #         if args.collect_all:
    #             y_pred, y_real = temporal_avg(y_pred, y_real, kernel_size, args.kind)
    #         else:
    #             y_pred = y_pred.cpu().numpy()
    #             y_real = y_real.cpu().numpy()

    #         m = calculate_metrics(y_pred, y_real, quantiles)
    #         avg_m.append(m)
    #         if i in [0, "t"]:
    #             print("plotting")
    #             plot_fcst(
    #                 y_pred,
    #                 y_real,
    #                 kernel_size=kernel_size,
    #                 save_name=f"assets/{args.dataset}_{args.pred_len}_{args.model_name}_{'fast_' if args.fast_sample else ''}{'dtm_' if args.deterministic else ''}{'multi_' if args.collect_all else ''}{d.split('/')[-1]}.png",
    #             )
    #         if args.smoke_test:
    #             break

    #     avg_m = np.array(avg_m)
    #     df = pd.DataFrame(
    #         avg_m.mean(axis=0, keepdims=False if args.collect_all else True),
    #         columns=["MAE", "MSE", "CRPS"],
    #         # columns=["RMSE", "MAE", "CRPS", "MPBL"],
    #     )
    #     df["method"] = d.split("/")[-1]
    #     df["granularity"] = kernel_size if args.collect_all else 1
    #     df_out.append(df)

    # df_out = pd.concat(df_out)
    # df_out = df_out.reindex(columns=["method", "granularity", "MAE", "MSE", "CRPS"])
    # print(df_out)
    # # df_out.to_csv("test.csv")
    # if not args.smoke_test:
    #     df_out.to_csv(
    #         f"assets/{args.dataset}_{args.pred_len}_{args.model_name}_{'fast' if args.fast_sample else ''}_{'dtm' if args.deterministic else ''}.csv"
    #     )


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
    args = parser.parse_args()
    main(args)
