import random
import pandas as pd
import torch
import numpy as np
import os
import glob
from src.models.diffusion import MovingAvgDiffusion
from src.utils.filters import get_factors
from src.utils.train import setup_seed
from src.utils.sample import Sampler, plot_fcst, temporal_avg
from src.utils.metrics import calculate_metrics
import matplotlib.pyplot as plt
from lightning import Trainer
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_seed()
print(device)


root_path = "/home/user/data/FrequencyDiffusion/savings/mfred"
# root_path = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings/mfred"
# dataset = "benchmarks"
dataset = "MovingAvgDiffusion"
deterministic = True
fast_sample = True
smoke_test = True
collect_all = True
n_sample = 200
num_training = 1

exp_path = os.path.join(root_path, dataset)

if fast_sample:
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


def main():
    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))

    # with open(os.path.join(root_path, "y_real.npy"), "rb") as f:
    #     y_real = np.load(f)

    with open(os.path.join(root_path, "scaler.npy"), "rb") as f:
        scaler = np.load(f, allow_pickle="TRUE").item()

    df_out = []
    exp_dirs = glob.glob(exp_path + "/*Backbone*")
    # exp_dirs = glob.glob("*Backbone*", root_dir=exp_path)
    # exp_dirs.sort()
    exp_dirs = [e[:-2] for e in exp_dirs]
    exp_dirs = list(set(exp_dirs))
    exp_dirs.sort()
    # print(exp_dirs)

    for d in exp_dirs:
        # for d in exp_dirs:
        avg_m = []
        for i in range(num_training):
            # i = 't'
            read_d = os.path.join(exp_path, d + f"_{i}", "diffusion.pt")
            print(read_d)
            kind = d.split("_")[1]
            diff = torch.load(read_d)

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
            if deterministic:
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
            diff.config_sample(sigmas, sample_steps)
            print(diff.sigmas)

            # PLOT WEIGHTS
            # cn = diff.conditioner
            # for name, p in cn.named_parameters():
            #     print(name, p[0])
            # if name.__contains__('weight'):
            #     fig, ax = plt.subplots()
            #     pos = ax.imshow(p.detach().cpu().numpy(), cmap='RdBu')
            #     cbar = fig.colorbar(pos, ax=ax)
            #     # cbar.minorticks_on()
            #     fig.tight_layout()
            #     fig.savefig('assets/'+name+'.png')

            sampler = Sampler(diff, n_sample, scaler)

            # y_pred, y_real = sampler.predict(test_dl)
            y_pred, y_real = sampler.sample(
                test_dl, smoke_test=smoke_test, collect_all=collect_all
            )
            # print(y_pred.shape)
            if collect_all:
                y_pred, y_real = temporal_avg(y_pred, y_real, kernel_size, kind)

            m = calculate_metrics(y_pred, y_real)
            avg_m.append(m)
            if i in [0, "t"]:
                plot_fcst(
                    y_pred,
                    y_real,
                    kernel_size=kernel_size,
                    save_name=f"assets/{'fast_' if fast_sample else ''}{'dtm_' if deterministic else ''}{'multi_' if collect_all else ''}{d.split('/')[-1]}.png",
                )

        avg_m = np.array(avg_m)
        df = pd.DataFrame(
            avg_m.mean(axis=0, keepdims=False if collect_all else True),
            columns=["RMSE", "MAE", "CRPS"],
        )
        df["method"] = d.split("/")[-1]
        df["granularity"] = kernel_size if collect_all else 1
        df_out.append(df)

    df_out = pd.concat(df_out)
    df_out = df_out.reindex(columns=["method", "granularity", "RMSE", "MAE", "CRPS"])
    # print(df_out)
    # df_out.to_csv("test.csv")
    df_out.to_csv(
        f"{'fast' if fast_sample else ''}_{'dtm' if deterministic else ''}.csv"
    )


if __name__ == "__main__":
    # main()
    trainer = Trainer(fast_dev_run=True, accelerator="gpu", devices=1)
    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))
    # ckpt = torch.load("lightning_logs/version_0/checkpoints/epoch=4-step=2800.ckpt")
    diff = MovingAvgDiffusion.load_from_checkpoint(
        "/home/user/data/FrequencyDiffusion/savings/mfred/MovingAvgDiffusion/MLPBackbone_freq_norm_True_diff_True_hot_/lightning_logs/version_0/checkpoints/epoch=2-step=1680.ckpt"
    )
    # diff = MovingAvgDiffusion.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=2-step=1680.ckpt")
    cn = diff.backbone
    for name, p in cn.named_parameters():
        print(name, p[0])
        break
    # bb = encoder_weights = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("backbone")}
    # print(bb)
    # # print(ckpt["state_dict"])
    # diff = MovingAvgDiffusion.load_from_checkpoint(
    #     "lightning_logs/version_4/checkpoints/epoch=4-step=2800.ckpt",
    #     hparams_file="lightning_logs/version_4/hparams.yaml",
    # )
    # diff.configure_sampling()
    # preds = trainer.predict(diff, dataloaders=test_dl)
    # preds = torch.stack(preds)
    # plt.plot(preds[0,0])
    # plt.savefig('assets/test.png')
    # print(preds.shape)
    # y_pred, y_real = temporal_avg(preds, y_real, kernel_size, kind)
