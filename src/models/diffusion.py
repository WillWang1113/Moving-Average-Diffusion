import abc
from typing import Dict
import torch
from torch import nn
from ..schedules.collection import linear_schedule
from ..utils.fourier import idft
from ..utils.filters import get_factors, MovingAvgTime, MovingAvgFreq
import matplotlib.pyplot as plt
from .conditioner import BaseConditioner


class BaseDiffusion(abc.ABC):
    @torch.no_grad
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @torch.no_grad
    def backward(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()

    def get_loss(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()

    def _encode_condition(self, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()

    def init_noise(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()


class DDPM(BaseDiffusion):
    def __init__(
        self,
        backbone: nn.Module,
        conditioner: BaseConditioner = None,
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
        c = self._encode_condition(condition)

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
        c = self._encode_condition(condition)
        eps_theta = self.backbone(x_noisy, t, c)

        # compute loss
        loss = nn.functional.mse_loss(eps_theta, noise)
        return loss

    def _encode_condition(self, condition: dict = None):
        if (condition is not None) and (self.conditioner is not None):
            c = self.conditioner(**condition)
        else:
            c = None
        return c

    def init_noise(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        return torch.randn_like(x, device=x.device)


class MovingAvgDiffusion(BaseDiffusion):
    """MovingAvgDiffusion. Limit to the average terms be the factors of the seq_length.

    Following Cold Diffusion

    Args:
        BaseDiffusion (_type_): _description_
    """

    def __init__(
        self,
        seq_length: int,
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

        # get int factors
        factors = get_factors(seq_length) + [seq_length]

        # get 2,3,..., seq_length
        # factors = [i for i in range(2, backbone.seq_length + 1)]
        self.T = len(factors)
        if freq_kw["frequency"]:
            # TODO: implement frequency response multiplication
            freq = torch.fft.rfftfreq(seq_length)
            self.degrade_fn = [MovingAvgFreq(f, freq=freq) for f in factors]
        else:
            self.degrade_fn = [MovingAvgTime(f) for f in factors]

    @torch.no_grad
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        unique_t, indexes = torch.unique(t, return_inverse=True, sorted=True)
        unique_idx = torch.unique(
            indexes,
            sorted=True,
        )
        x_noisy = torch.empty_like(x, device=x.device)
        for i in unique_idx:
            sub_t = unique_t[i]
            degrade_fn = self.degrade_fn[sub_t]
            x_noisy[indexes == i] = degrade_fn(x[indexes == i])
        return x_noisy

    @torch.no_grad
    def backward(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        c = self._encode_condition(condition)

        x_ts = [x.clone()]
        for t in range(self.T - 1, -1, -1):
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])
            x_hat = self.backbone(x, t_tensor, c)

            if t == 0:
                # x = self.degrade_fn[t - 1](x_hat)
                # x = x_hat
                x = x - self.degrade_fn[t](x_hat) + x_hat
            else:
                x = x - self.degrade_fn[t](x_hat) + self.degrade_fn[t - 1](x_hat)

            # fig, ax = plt.subplots()
            # ax.plot(x[0].detach())
            # fig.suptitle(f"{t}")
            # fig.savefig(f"assets/denoise_{t}.png")
            # plt.close()
            x_ts.append(x)

        if self.freq_kw["frequency"]:
            x_ts = [idft(x_t, self.freq_kw["stereographic"]) for x_t in x_ts]

        return torch.stack(x_ts, dim=-1)

    def get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy = self.forward(x, t)

        # look at corrupted data
        # fig, ax = plt.subplots()
        # ax.plot(x_noisy[0].detach())
        # fig.suptitle(f'{t[0].detach()}')
        # fig.savefig(f'assets/noisy_{t[0]}.png')
        # plt.close()

        # eps_theta
        c = self._encode_condition(condition)
        x_hat = self.backbone(x_noisy, t, c)

        # compute loss
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def init_noise(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        # TODO: condition on basic predicion f(x)
        if (condition is not None) and (self.conditioner is not None):
            # ! just for test !
            # ! use observed last !
            if self.freq_kw["frequency"]:
                prev_value = condition["observed_data"][:, [0], :]
                prev_value = prev_value + torch.randn(1, device=x.device) * prev_value.max() * 0.1
                noise = torch.concat([prev_value, torch.zeros(x.shape[0], x.shape[1]-1, x.shape[2], device=x.device)], dim=1)
                
                
            else:
                prev_value = condition["observed_data"][:, [-1], :]
                prev_value = prev_value + torch.randn(1, device=x.device) * prev_value.max() * 0.1
                noise = prev_value.expand_as(x)

            # noise = self.forward(
            #     x,
            #     torch.ones((x.shape[0],), device=x.device, dtype=torch.int)
            #     * (self.T - 1),
            # )

        else:
            # condition on given target
            noise = self.forward(
                x,
                torch.ones((x.shape[0],), device=x.device, dtype=torch.int)
                * (self.T - 1),
            )

            # # zero-mean if data is pre-normalized?
            # noise = torch.zeros_like(x, device=x.device)

        # add a small amount of noise
        # noise = noise + torch.randn_like(noise, 10) * 5e-2
        return noise

    def _encode_condition(self, condition: dict = None):
        if (condition is not None) and (self.conditioner is not None):
            c = self.conditioner(**condition)
        else:
            c = None
        return c
