import sys
import torch
import tqdm
import os
from src.utils.filters import MovingAvgTime, MovingAvgFreq

thismodule = sys.modules[__name__]


def get_schedule(noise_name, n_steps, data_name=None,train_dl=None, check_pth=None):
    if noise_name != "std":
        assert noise_name in ["linear", "cosine", "zero", "freqresp"]
        fn = getattr(thismodule, noise_name + "_schedule")
        return fn(n_steps)

    else:
        assert (check_pth is not None) and (train_dl is not None)
        return std_schedule(data_name, train_dl, check_pth)


def linear_schedule(n_steps, min_beta=1e-4, max_beta=2e-2):
    betas = torch.linspace(min_beta, max_beta, n_steps)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    return {
        "alpha_bars": alpha_bars.float(),
        "beta_bars": None,
        "alphas": alphas.float(),
        "betas": betas.float(),
    }
    # return alpha_bars, beta_bars, alphas, betas

# def linear_schedule(n_steps, min_beta=1e-4, max_beta=2e-2):
#     beta_bars = torch.linspace(min_beta, max_beta, n_steps)
#     alpha_bars = 1 - beta_bars
#     alphas = torch.cumprod(alpha_bars, dim=0)
#     betas = torch.sqrt(1 - alpha_bars)
#     return {
#         "alpha_bars": alpha_bars.float(),
#         "beta_bars": beta_bars.float(),
#         "alphas": alphas.float(),
#         "betas": betas.float(),
#     }
#     return alpha_bars, beta_bars, alphas, betas


def cosine_schedule(n_steps):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = n_steps + 1  # Number of steps is one more than timesteps
    x = torch.linspace(0, n_steps, steps)  # Create a linear space from 0 to timesteps
    # Calculate the cumulative product of alphas using the cosine schedule formula
    alphas_cumprod = (
        torch.cos(((x / n_steps) + 8e-3) / (1 + 8e-3) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = (
        alphas_cumprod / alphas_cumprod[0]
    )  # Normalize by the first element
    betas = 1 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )  # Calculate betas from alphas
    betas = torch.clip(betas, 0.0001, 0.9999)
    alphas = 1 - betas
    return {
        "alpha_bars": alphas.float(),
        "beta_bars": betas.float(),
        "alphas": alphas_cumprod[1:].float(),
        "betas": torch.sqrt(1 - alphas_cumprod[1:]).float(),
    }

    return (
        alphas,
        betas,
        alphas_cumprod[1:],
        torch.sqrt(1 - alphas_cumprod[1:]),
    )  # Clip betas to be within specified range


def zero_schedule(n_steps):
    return {
        "alpha_bars": None,
        "beta_bars": None,
        "alphas": torch.ones(n_steps).float(),
        "betas": torch.zeros(n_steps).float(),
    }


def std_schedule(data_name, train_dl, check_pth):
    batch = next(iter(train_dl))
    seq_length = batch["future_data"].shape[1]

    file_name = os.path.join(check_pth, f"norm_stdratio_{data_name}_{seq_length}.pt")
    exist = os.path.exists(file_name)
    if exist:
        print("Use pre-computed schedule!")
        all_ratio = torch.load(file_name)
        return {
            "alpha_bars": None,
            "beta_bars": None,
            "alphas": None,
            "betas": torch.sqrt(1 - all_ratio**2).float(),
        }
        return torch.sqrt(1 - all_ratio**2)
    else:
        print("Calculate schedule...")
        batch = next(iter(train_dl))
        seq_length = batch["future_data"].shape[1]

        all_ratio = []
        for batch in tqdm.tqdm(train_dl):
            x = batch["future_data"]
            mean = torch.mean(x, dim=1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-6)
            x_norm = (x - mean) / stdev
            
            
            orig_std = 1
            
            step_ratios = []
            for j in range(2, seq_length + 1):
                mat = MovingAvgTime(j, seq_length=seq_length)
                x_avg_norm = mat(x_norm)
                std_avg = torch.sqrt(
                    torch.var(x_avg_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                ratio = std_avg / orig_std
                step_ratios.append(ratio)
            step_ratios = torch.stack(step_ratios)
            all_ratio.append(step_ratios)
        all_ratio = torch.concat(all_ratio, dim=1).mean(dim=1).squeeze()
        torch.save(all_ratio, file_name)
        return {
            "alpha_bars": None,
            "beta_bars": None,
            "alphas": None,
            "betas": torch.sqrt(1 - all_ratio**2).float(),
        }
        return all_ratio, torch.sqrt(1 - all_ratio**2)


def freqresp_schedule(n_steps):
    seq_len = n_steps + 1
    fr = [MovingAvgFreq(i, seq_len).K.diag() for i in range(2, seq_len + 1)]
    fr = torch.stack(fr)  # [steps, seq_len//2 + 1]
    fr = 1 - (fr.conj() * fr).real
    fr = torch.sqrt(fr + 1e-6)
    fr[:, 0] = torch.ones_like(fr[:, 0]) * 1e-5
    # fr_im = fr.clone()
    # fr_im = fr_im[:, 1:]
    # if seq_len % 2 == 0:
    #     fr_im = fr_im[:, :-1]
    # fr_cat = torch.cat([fr, fr_im], dim=1)  # [steps, seq_len]
    # assert fr_cat.shape[0] == n_steps
    # assert fr_cat.shape[1] == seq_len
    return {
        "alpha_bars": None,
        "beta_bars": None,
        "alphas": None,
        "betas": fr.float(),
    }
