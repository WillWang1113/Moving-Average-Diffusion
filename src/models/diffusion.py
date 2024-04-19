import abc
from typing import Dict
import torch
from torch import nn
from ..schedules.collection import linear_schedule
from ..utils.fourier import idft
from ..utils.misc import get_factors, MovingAvg
import matplotlib.pyplot as plt


class BaseDiffusion(abc.ABC):
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @torch.no_grad
    def backward(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()

    def get_loss(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()

    def encode_condition(self, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()


class DDPM(BaseDiffusion):
    def __init__(
        self,
        backbone: nn.Module,
        conditioner: nn.Module = None,
        freq_kw={"frequency": False, "stereographic": False},
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
        self.freq_kw = freq_kw

    @torch.no_grad
    def forward(self, x, t, noise):
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy

    @torch.no_grad
    def backward(self, x, condition):
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

        if self.freq_kw["frequency"]:
            x = idft(x, self.freq_kw["stereographic"])

        return x

    def get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)
        noise = torch.randn_like(x).to(self.device)

        # corrupt data
        x_noisy = self.forward(x, t, noise)

        # look at corrupted data
        # fig, ax = plt.subplots()
        # ax.plot(x_noisy[0].detach())
        # fig.suptitle(f'{t[0].detach()}')
        # fig.savefig(f'assets/noisy_{t[0]}.png')
        # plt.close()

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


class MovingAvgDiffusion(DDPM):
    """MovingAvgDiffusion. Limit to the average terms be the factors of the seq_length.

    Following Cold Diffusion

    Args:
        BaseDiffusion (_type_): _description_
    """

    def __init__(
        self,
        backbone: nn.Module,
        conditioner: nn.Module = None,
        freq_kw={"frequency": False, "stereographic": False},
        device="cpu",
    ) -> None:
        self.backbone = backbone.to(device)
        self.conditioner = conditioner
        if conditioner is not None:
            self.conditioner = self.conditioner.to(device)
            print("Have conditioner")
        self.device = device
        self.freq_kw = freq_kw
        factors = get_factors(backbone.seq_length)
        self.T = len(factors)
        if freq_kw["frequency"]:
            # TODO: implement frequency response multiplication
            self.degrade_fn = None
        else:
            self.degrade_fn = [MovingAvg(f) for f in factors]

    @torch.no_grad
    def forward(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        unique_t, indexes = torch.unique(t, return_inverse=True, sorted=True)
        unique_idx = torch.unique(
            indexes,
            sorted=True,
        )
        for i in unique_idx:
            sub_t = unique_t[i]
            degrade_fn = self.degrade_fn[sub_t]
            x[indexes == i] = degrade_fn(x[indexes == i])
        return x

    @torch.no_grad
    def backward(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        c = self.encode_condition(condition)

        # introduce a little noise
        # x = x + torch.randn_like(x) * 1e-4

        for t in range(self.T - 1, 0, -1):
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])
            x_hat = self.backbone(x, t_tensor, c)
            x = x - self.degrade_fn[t](x_hat) + self.degrade_fn[t - 1](x_hat)

        if self.freq_kw["frequency"]:
            x = idft(x, self.freq_kw["stereographic"])

        return x

    def get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        # Hot diff
        # noise = torch.randn_like(x).to(self.device)
        # cold diff
        noise = None

        # corrupt data
        x_noisy = self.forward(x, t, noise)

        # look at corrupted data
        # fig, ax = plt.subplots()
        # ax.plot(x_hat[0].detach())
        # fig.suptitle(f'{t[0].detach()}')
        # fig.savefig(f'assets/noisy_{t[0]}.png')
        # plt.close()

        # eps_theta
        c = self.encode_condition(condition)
        x_hat = self.backbone(x_noisy, t, c)
        

        # compute loss
        loss = nn.functional.mse_loss(x_hat, x)
        return loss
