import random
import pandas as pd
import torch
import numpy as np
import os
import glob
from src.utils.filters import get_factors
from src.utils.train import setup_seed
from src.utils.sample import Sampler
from src.utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_seed()
print(device)


root_path = "/home/user/data/FrequencyDiffusion/savings/mfred"
# dataset = "benchmarks"
dataset = "MovingAvgDiffusion"
deterministic = True
fast_sample = True
smoke_test = True
collect_all = True

exp_path = os.path.join(root_path, dataset)
num_training = 1

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
kernel_size+=[1]
print(kernel_size)

def main():
    test_dl = torch.load(os.path.join(root_path, "test_dl.pt"))

    with open(os.path.join(root_path, "y_real.npy"), "rb") as f:
        y_real = np.load(f)

    with open(os.path.join(root_path, "scaler.npy"), "rb") as f:
        scaler = np.load(f, allow_pickle="TRUE").item()

    df_out = {}
    exp_dirs = glob.glob("MLPBackbone_*", root_dir=exp_path)
    # exp_dirs.sort()
    exp_dirs = [e[:-2] for e in exp_dirs]
    exp_dirs = list(set(exp_dirs))
    exp_dirs.sort()
    print(exp_dirs)

    for d in exp_dirs:
        # for d in exp_dirs:
        avg_m = []
        for i in range(num_training):
            read_d = os.path.join(exp_path, d + f"_{i}", "diffusion.pt")
            print(read_d)
            kind = d.split('_')[1]
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
                sigmas = torch.linspace(1.5e-3, 0.5, diff.T, device=diff.device)
            diff.config_sample(sigmas, sample_steps)

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

            sampler = Sampler(diff, 50, scaler)

            # y_pred, y_real = sampler.predict(test_dl)
            y_pred, y_real = sampler.sample(
                test_dl, smoke_test=smoke_test, collect_all=collect_all
            )
            print(y_pred.shape)

            m = calculate_metrics(y_pred, y_real, kind=kind, kernel_size=kernel_size)
            avg_m.append(m)
            # bs, ts, dims = y_pred.shape
            n_sample = y_pred.shape[0]
            bs = y_pred.shape[1]

            if i == 0 or i == "t":
                fig, ax = plt.subplots(3, 3)
                ax = ax.flatten()
                for k in range(9):
                    choose = np.random.randint(0, bs)
                    sample_real = y_real[choose, :, 0]
                    # sample_pred = y_pred[choose, :, 0]
                    sample_pred = y_pred[:, choose, :, 0].T
                    # ax[k].scatter(sample_pred[:,2], sample_pred[:,5])
                    ax[k].plot(sample_real, label="real")
                    ax[k].legend()
                    # ax[k].plot(sample_pred, c="black")
                    ax[k].plot(sample_pred, c="black", alpha=1 / n_sample)
                # fig.suptitle(f"ma_term: {L}")
                fig.tight_layout()
                fig.savefig(
                    f"assets/{'fast' if fast_sample else ''}_{'dtm' if deterministic else 'sto'}_{d}.png"
                )
        avg_m = np.array(avg_m)
        df_out[d] = avg_m.mean(axis=0)
        print(df_out[d])
        break
    # df_out = pd.DataFrame(df_out, index=["RMSE", "MAE", "CRPS"]).T
    # print(df_out)
    # df_out.to_csv(f"{'fast' if fast_sample else ''}_{'dtm' if deterministic else ''}.csv")


if __name__ == "__main__":
    main()
