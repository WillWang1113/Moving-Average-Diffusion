import torch


def noise_schedule(noise_kw: dict):
    noise_kw_copy = noise_kw.copy()

    if noise_kw_copy.pop("name") == "linear":
        return linear_schedule(**noise_kw_copy)
    elif noise_kw_copy.pop("name") == "cosine":
        return cosine_schedule(**noise_kw_copy)
    else:
        raise ValueError()


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