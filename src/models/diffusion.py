import torch
from src.schedules.exponential import exponential_schedule


class DDPM(object):
    def __init__(self, T, min_beta, max_beta, device="cpu") -> None:
        alphas, betas, alpha_bars = exponential_schedule(min_beta, max_beta, T, device)
        self.alphas = alphas
        self.betas = betas
        self.alpha_bars = alpha_bars
        self.T = T

    def forward(self, x, t, noise):
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy

    @torch.no_grad
    def backward(self, noise, backbone):
        x = noise
        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(x)
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])
            noise_pred = backbone(x, t_tensor)
            mu_pred = (
                x - (1 - self.alphas[t]) / (torch.sqrt(self.alpha_bars[t])) * noise_pred
            )
            if t == 0:
                sigma = 0
            else:
                sigma = torch.sqrt(
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
            x = mu_pred + sigma * z
        return x
