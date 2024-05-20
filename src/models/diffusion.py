import abc
from typing import Dict
import torch
from torch import nn

from src.models.blocks import RevIN
from ..utils import schedule
from ..utils.fourier import (
    complex_freq_to_real_imag,
    dft,
    idft,
    real_imag_to_complex_freq,
)
from ..utils.filters import get_factors, MovingAvgTime, MovingAvgFreq
from .conditioner import BaseConditioner
import matplotlib.pyplot as plt


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

    def get_params(self):
        raise NotImplementedError()


class DDPM(BaseDiffusion):
    def __init__(
        self,
        backbone: nn.Module,
        conditioner: BaseConditioner = None,
        freq_kw={"frequency": False},
        T=100,
        noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
        device="cpu",
        **kwargs,
    ) -> None:
        alphas, betas, alpha_bars, _ = getattr(
            schedule, noise_kw["name"] + "_schedule"
        )(self.T, device=device, **noise_kw)
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
        # extend condition into n_sample * batchsize
        c = self._encode_condition(condition)
        if c.shape[0] != x.shape[0]:
            c = c.repeat(x.shape[0] // c.shape[0], 1)

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
            x = idft(x)

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

    def init_noise(
        self,
        x: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
        n_sample: int = 1,
    ):
        shape = (n_sample, x.shape[0], x.shape[1], x.shape[2])
        return torch.randn(shape, device=x.device).flatten(end_dim=1)

    def get_params(self):
        params = list(self.backbone.parameters())
        if self.conditioner is not None:
            params += list(self.conditioner.parameters())
        return params


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
        freq_kw={"frequency": False},
        device="cpu",
        noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
        only_factor_step=False,
        norm=True,
        fit_on_diff=False,
        mode="VE",
    ) -> None:
        self.backbone = backbone.to(device)
        self.conditioner = conditioner
        if conditioner is not None:
            self.conditioner = self.conditioner.to(device)
            print("CONDITIONAL DIFFUSION!")
        self.device = device
        self.freq_kw = freq_kw
        self.seq_length = seq_length

        # get int factors
        middle_res = get_factors(seq_length) + [seq_length]

        # get 2,3,..., seq_length
        factors = (
            middle_res if only_factor_step else [i for i in range(2, seq_length + 1)]
        )
        self.T = len(factors)

        if freq_kw["frequency"]:
            print("DIFFUSION IN FREQUENCY DOMAIN!")
            freq = torch.fft.rfftfreq(seq_length)
            self.degrade_fn = [MovingAvgFreq(f, freq=freq) for f in factors]
            self.freq_response = torch.concat(
                [df.Hw.to(self.device) for df in self.degrade_fn]
            )
        else:
            print("DIFFUSION IN TIME DOMAIN!")
            self.degrade_fn = [MovingAvgTime(f) for f in factors]

        noise_schedule = getattr(schedule, noise_kw.pop("name") + "_schedule")(
            n_steps=self.T, device=device, **noise_kw
        )
        # for corruption
        self.betas = noise_schedule[-1]
        self.alphas = noise_schedule[-2]
        if torch.allclose(self.betas, torch.zeros_like(self.betas)):
            print("COLD DIFFUSION")
            self.cold = True
        else:
            print("HOT DIFFUSION")
            self.cold = False

        self.norm = norm
        self.fit_on_diff = fit_on_diff
        self.mode = mode
        assert mode in ["VE", "VP"]

    # TODO: too slow
    @torch.no_grad
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        alpha_ = 1.0 if self.mode == "VE" else self.alphas[t].unsqueeze(1).unsqueeze(1)

        if self.freq_kw["frequency"]:
            x_complex = dft(x, real_imag=False)
            # # real+imag --> complex tensor
            # x_complex = real_imag_to_complex_freq(x) if self.freq_kw["real_imag"] else x

            # matrix multiplication
            x_filtered = x_complex * self.freq_response[t]

            # add noise if needed
            x_filtered = alpha_ * x_filtered + self.betas[t].unsqueeze(1).unsqueeze(
                1
            ) * torch.randn_like(x_filtered).to(self.device)

            # complex tensor --> real+imag
            x_noisy = (
                complex_freq_to_real_imag(x_filtered)
                if self.freq_kw["real_imag"]
                else x_filtered
            )
        else:
            # loop for each sample to temporal average
            x_noisy = []
            for i in range(x.shape[0]):
                x_n = self.degrade_fn[t[i]](x[[i], ...])
                x_noisy.append(x_n)
            x_noisy = torch.concat(x_noisy)
            x_noisy = x_noisy * alpha_ + self.betas[t].unsqueeze(1).unsqueeze(
                1
            ) * torch.randn_like(x_noisy).to(self.device)

        return x_noisy

    @torch.no_grad
    def backward(
        self,
        x: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
        collect_all: bool = False,
    ):
        c, (mu, std) = self._encode_condition(condition)
        if c.shape[0] != x.shape[0]:
            c = c.repeat(x.shape[0] // c.shape[0], 1)
            mu = mu.repeat(x.shape[0] // mu.shape[0], 1, 1)
            std = std.repeat(x.shape[0] // std.shape[0], 1, 1)

        # get denoise steps (DDIM/full)
        iter_T = self.sample_Ts
        # iter_T = self.sample_Ts if self.fast_sample else list(range(self.T - 1, -1, -1))

        all_x = [x]
        for i in range(len(iter_T)):
            # i is the index of denoise step, t is the denoise step
            t = iter_T[i]
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])

            # TODO: correct normalize when backward!
            if self.norm:
                mean = torch.mean(x, dim=1, keepdim=True)
                stdev = torch.sqrt(
                    torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                x_norm = (x - mean) / stdev
            else:
                x_norm = x

            if self.freq_kw["frequency"]:
                x_norm = dft(x_norm, **self.freq_kw)

            # estimate x_0
            x_hat = self.backbone(x_norm, t_tensor, c)
            if self.fit_on_diff:
                x_hat = x_hat + x_norm

            # in frequency domain, it's easier to do iteration in complex tensor
            if self.freq_kw["frequency"]:
                x = dft(x)
                # x = real_imag_to_complex_freq(x) if self.freq_kw["real_imag"] else x
                x_hat = (
                    real_imag_to_complex_freq(x_hat)
                    if self.freq_kw["real_imag"]
                    else x_hat
                )

            if t == 0:
                x = x_hat + torch.randn_like(x) * self.sigmas[t]
            else:
                # calculate coeffs
                prev_t = iter_T[i + 1]
                if not self.cold:
                    coeff = self.betas[prev_t] ** 2 - self.sigmas[t] ** 2
                    coeff = torch.sqrt(coeff) / self.betas[t]
                    sigma = self.sigmas[t]
                else:
                    coeff = 1
                    sigma = 0
                x = (
                    x * coeff
                    + self.degrade_fn[prev_t](x_hat)
                    - self.degrade_fn[t](x_hat) * coeff
                )

                # fig, ax = plt.subplots()
                # if self.freq_kw['frequency']:
                #     x_plot = idft(x, real_imag=False)
                # else:
                #     x_plot = x.clone()
                # for ii in range(3):
                #     ax.plot(x_plot[ii].detach().cpu().real.numpy(), alpha=0.5)

                # fig.suptitle(f"{t}")
                # fig.savefig(f"assets/noisy_{t}.png")
                # plt.close()

                x = x + torch.randn_like(x) * sigma

            # in frequency domain, backbone use real+imag to train
            if self.freq_kw["frequency"]:
                x = idft(x, real_imag=False)
                # x_hat = idft(x_hat, **self.freq_kw)
            if collect_all:
                all_x.append(x * std + mu)

        if collect_all:
            all_x = torch.stack(all_x, dim=-1)
            # x_ts = [idft(x_t) for x_t in x_ts]
        return x * std + mu if not collect_all else all_x
        # return torch.stack(x_ts, dim=-1)

    def get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        if self.norm:
            mean = torch.mean(x, dim=1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        else:
            mean, stdev = (
                torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device),
                torch.ones((x.shape[0], 1, x.shape[-1]), device=x.device),
            )
        x_norm = (x - mean) / stdev


        # corrupt data
        x_noisy = self.forward(x_norm, t)

        # # look at corrupted data
        # fig, ax = plt.subplots()
        # ax.plot(x[0].detach().cpu())
        # ax.plot(x_noisy[0].detach().cpu())
        # fig.suptitle(f'{t[0].detach().cpu()}')
        # fig.savefig(f'assets/noisy_{t[0]}.png')
        # plt.close()

        # eps_theta
        # GET mu and std in time domain
        c, (mu, std) = self._encode_condition(condition)
        x_hat = self.backbone(x_noisy, t, c)
        if self.fit_on_diff:
            x_hat = x_hat + x_noisy
        if self.freq_kw['frequency']:
            x_hat = x_hat * std 
            x_hat[:,[0],:] = x_hat[:,[0],:] + mu * self.seq_length**0.5
        else:
            x_hat = x_hat * std + mu

        # compute loss
        if self.freq_kw["frequency"]:
            x = dft(x, **self.freq_kw)

        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def init_noise(
        self,
        x: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
        n_sample: int = 1,
    ):
        # * INIT ON TIME DOMAIN AND DO TRANSFORM
        # TODO: condition on basic predicion f(x)
        if (condition is not None) and (self.conditioner is not None):
            # ! just for test !
            # ! use latest observed as zero frequency component !
            time_value = condition["observed_data"][:, [-1], :]
            prev_value = torch.zeros_like(time_value) if self.norm else time_value
            prev_value = prev_value.expand(-1, x.shape[1], -1)

            noise = (
                prev_value
                + torch.randn(
                    n_sample,
                    *prev_value.shape,
                    device=prev_value.device,
                    dtype=prev_value.dtype,
                )
                * self.betas[-1]
            ).flatten(end_dim=1)

        else:
            # condition on given target
            noise = self.forward(
                x,
                torch.ones((x.shape[0],), device=x.device, dtype=torch.int)
                * (self.T - 1),
            )

        return noise

    def _encode_condition(self, condition: dict = None):
        if (condition is not None) and (self.conditioner is not None):
            c, (mu, std) = self.conditioner(**condition)
        else:
            c = None
            mu, std = None, None
        return c, (mu, std)

    def get_params(self):
        params = list(self.backbone.parameters())
        if self.conditioner is not None:
            params += list(self.conditioner.parameters())
        return params

    def config_sample(self, sigmas=None, sample_steps=None):
        self.sigmas = sigmas
        self.sample_Ts = (
            list(range(self.T - 1, -1, -1)) if sample_steps is None else sample_steps
        )
        for i in range(len(self.sample_Ts) - 1):
            t = self.sample_Ts[i]
            prev_t = self.sample_Ts[i + 1]
            assert self.betas[prev_t] >= self.sigmas[t]
