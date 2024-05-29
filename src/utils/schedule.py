import torch
import tqdm

from src.utils.filters import MovingAvgTime


def linear_schedule(min_beta, max_beta, n_steps):
    betas = torch.linspace(min_beta, max_beta, n_steps)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, betas, alpha_bars, torch.sqrt(1 - alpha_bars)


def cosine_schedule(min_beta, max_beta, n_steps):
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
    return (
        alphas,
        betas,
        alphas_cumprod[1:],
        torch.sqrt(1 - alphas_cumprod[1:]),
    )  # Clip betas to be within specified range


def variance_schedule(train_dl):
    batch = next(iter(train_dl))
    seq_length = batch["observed_data"].shape[1]

    all_ratio = []
    for batch in tqdm.tqdm(train_dl):
        x = batch["future_data"]
        orig_std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        step_ratios = []
        for j in range(2, seq_length + 1):
            mat = MovingAvgTime(j)
            x_avg = mat(x)
            std_avg = torch.sqrt(
                torch.var(x_avg, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            ratio = std_avg / orig_std
            step_ratios.append(ratio)
        step_ratios = torch.stack(step_ratios)
        all_ratio.append(step_ratios)
    all_ratio = torch.concat(all_ratio, dim=1).mean(dim=1)
    return all_ratio.squeeze()
