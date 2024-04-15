from typing import Dict
import torch
from torch import nn
from ..schedules.collection import linear_schedule
from ..utils.fourier import idft


class DDPM(object):
    def __init__(
        self,
        backbone: nn.Module,
        conditioner: nn.Module = None,
        frequency=False,
        T=100,
        min_beta=1e-4,
        max_beta=2e-2,
        device="cpu",
    ) -> None:
        alphas, betas, alpha_bars = linear_schedule(min_beta, max_beta, T, device)
        self.backbone = backbone.to(device)
        self.conditioner = conditioner
        if conditioner is not None:
            self.conditioner = self.conditioner.to(device)
            print("Have conditioner")

        self.alphas = alphas
        self.betas = betas
        self.alpha_bars = alpha_bars
        self.T = T
        self.device = device
        self.frequency = frequency

    def forward(self, x, t, noise):
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy

    @torch.no_grad
    def backward(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        """Reverse diffusion. From noise to data.

        Args:
            x (torch.Tensor): Gaussian noise
            condition (Dict[torch.Tensor], optional): condtions like label/previous data. Defaults to None.

        Returns:
            torch.Tensor: denoised x with same shape
        """
        c = self.encode_condition(condition)

        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(x)
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])
            noise_pred = self.backbone(x, t_tensor, c)
            mu_pred = (
                x
                - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * noise_pred
            ) / torch.sqrt(self.alphas[t])
            if t == 0:
                sigma = 0
            else:
                sigma = torch.sqrt(
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
            x = mu_pred + sigma * z

        if self.frequency:
            x = idft(x)

        return x

    def get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)
        noise = torch.randn_like(x).to(self.device)

        # corrupt data
        x_noisy = self.forward(x, t, noise)

        # eps_theta
        c = self.encode_condition(condition)
        eps_theta = self.backbone(x_noisy, t, c)

        # compute loss
        loss = nn.functional.mse_loss(eps_theta, noise)
        return loss

    def encode_condition(self, condition: dict = None):
        if (condition is not None) and (self.conditioner is not None):
            c = self.conditioner(**condition)
        else:
            c = None
        return c
