import math
import sys
import torch
import tqdm
import os
from src.utils.filters import get_factors
from src.utils.filters import MovingAvgTime

thismodule = sys.modules[__name__]


def get_schedule(noise_name, n_steps, check_pth, **kwargs):
    file_name = os.path.join(
        check_pth,
        f"{noise_name}_sched_{n_steps}.pt",
    )
    exist = os.path.exists(file_name)
    if exist:
        print("Already exist!")
    else:
        os.makedirs(check_pth, exist_ok=True)
        if noise_name in ["linear", "cosine", "zero"]:
            fn = getattr(thismodule, noise_name + "_schedule")
            noise_schedule = fn(n_steps)
            torch.save(noise_schedule, file_name)

        elif noise_name == "std":
            assert kwargs.get("train_dl") is not None
            fn = getattr(thismodule, noise_name + "_schedule")
            noise_schedule = fn(n_steps, kwargs.get("train_dl"))
            torch.save(noise_schedule, file_name)
    return file_name




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



def cosine_schedule(n_steps: int, s: float = 0.008):
    """Cosine schedule for noise schedule.

    Args:
        n_steps (int): total number of steps.
        s (float, optional): tolerance. Defaults to 0.008.

    Returns:
        Dict[str, torch.Tensor]: noise schedule.
    """
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.001, 0.999)
    alphas = 1.0 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    noise_schedule = dict(
        betas=betas,
        alphas=alphas,
        alpha_bars=alphas_bars,
        beta_bars=None,
    )
    return noise_schedule


def zero_schedule(n_steps):
    return {
        "alpha_bars": None,
        "beta_bars": None,
        "alphas": torch.ones(n_steps).float(),
        "betas": torch.zeros(n_steps).float(),
    }


def std_schedule(n_steps, train_dl):
    batch = next(iter(train_dl))
    seq_length = batch["x"].shape[1]
    all_x = torch.concat([batch["x"] for batch in train_dl])

    orig_std = torch.sqrt(torch.var(all_x, dim=1, unbiased=False) + 1e-5)
    kernel_list = get_factors(seq_length)
    step_ratios, all_K = [], []
    all_K = torch.stack(
        [MovingAvgTime(j, seq_length=seq_length, stride=j).K for j in kernel_list]
    )
    all_K = (
        all_K.flatten(1).unsqueeze(0).permute(0, 2, 1)
    )  # [1, seq_len*seq_len, n_factors]
    interp_all_K = (
        torch.nn.functional.interpolate(
            all_K, size=n_steps + 1, mode="linear", align_corners=True
        )
        .squeeze()
        .reshape(seq_length, seq_length, -1)
        .permute(2, 0, 1)
    )  # [n_steps + 1, seq_len, seq_len]

    # first step is original data
    interp_all_K = interp_all_K[1:]

    for j in tqdm.tqdm(range(len(interp_all_K))):
        x_avg = interp_all_K[j] @ all_x
        std_avg = torch.sqrt(torch.var(x_avg, dim=1, unbiased=False) + 1e-5)
        ratio = std_avg / orig_std
        step_ratios.append(ratio.mean(dim=0))
    all_ratio = torch.stack(step_ratios)  # [n_factors, seq_chnl]

    print(interp_all_K.shape)
    print(torch.sqrt(1 - all_ratio**2).float().shape)
    return {
        "alpha_bars": None,
        "beta_bars": None,
        "alphas": interp_all_K,
        "betas": torch.sqrt(1 - all_ratio**2).float(),
    }


