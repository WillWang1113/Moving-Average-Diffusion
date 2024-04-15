import torch

def linear_schedule(min_beta, max_beta, n_steps, device):
    betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, betas, alpha_bars
    