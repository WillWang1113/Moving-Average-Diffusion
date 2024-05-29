from typing import Dict
import torch
from torch import nn

from ..models.backbone import build_backbone
from ..models.base import BaseDiffusion
from ..models.conditioner import build_conditioner
from ..utils import schedule
from ..utils.fourier import (
    complex_freq_to_real_imag,
    dft,
    idft,
    real_imag_to_complex_freq,
)
from ..utils.filters import get_factors, MovingAvgTime, MovingAvgFreq


class DDPM(BaseDiffusion):
    def __init__(
        self,
        backbone_config: dict,
        conditioner_config: dict,
        frequency: bool = False,
        # freq_kw={"frequency": False},
        T=100,
        noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
        **kwargs,
    ) -> None:
        super().__init__()
        alphas, betas, alpha_bars, _ = getattr(
            schedule, noise_kw["name"] + "_schedule"
        )(self.T, **noise_kw)
        self.backbone = build_backbone(backbone_config)
        self.conditioner = build_conditioner(conditioner_config)
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

        # self.alphas = alphas
        # self.betas = betas
        # self.alpha_bars = alpha_bars
        self.T = T
        self.freq = frequency

    @torch.no_grad
    def degrade(self, x, t):
        noise = torch.randn_like(x)
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy, noise

    def train_step(self, batch):
        x = batch["future_data"]
        condition = batch["condition"]
        loss = self._get_loss(x, condition)
        return loss

    @torch.no_grad
    def validation_step(self, batch):
        x = batch["future_data"]
        condition = batch["condition"]
        loss = self._get_loss(x, condition)
        return loss

    @torch.no_grad
    def predict_step(self, batch):
        target = batch["future_data"]
        condition = batch["condition"]
        x = self._init_noise(target)

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

        if self.freq:
            x = idft(x)

        return x

    def _get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy, noise = self.degrade(x, t)

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

    def _init_noise(
        self,
        x: torch.Tensor,
    ):
        shape = (self.n_sample, x.shape[0], x.shape[1], x.shape[2])
        return torch.randn(shape, device=x.device).flatten(end_dim=1)

    def configure_sampling(self, n_sample):
        self.n_sample = n_sample


class MADTime(BaseDiffusion):
    """MovingAvgDiffusionTime."""

    def __init__(
        self,
        backbone_config: dict,
        conditioner_config: dict,
        noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
        norm=True,
        pred_diff=False,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_config)
        self.conditioner = build_conditioner(conditioner_config)
        self.seq_length = self.backbone.seq_length
        self.norm = norm
        self.pred_diff = pred_diff

        # config diffusion steps
        self.factors = [i for i in range(2, self.seq_length + 1)]
        self.T = len(self.factors)
        self.degrade_fn = [MovingAvgTime(f) for f in self.factors]
        ns = noise_kw.pop("name")
        noise_schedule = getattr(schedule, ns + "_schedule")(n_steps=self.T, **noise_kw)
        self.register_buffer("betas", noise_schedule[-1])

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        # loop for each sample to temporal average
        x_noisy = []
        for i in range(x.shape[0]):
            x_n = self.degrade_fn[t[i]](x[[i], ...])
            x_noisy.append(x_n)
        x_noisy = torch.concat(x_noisy)
        x_noisy = x_noisy + self.betas[t].unsqueeze(1).unsqueeze(1) * torch.randn_like(
            x_noisy
        )
        return x_noisy

    def train_step(self, batch):
        x = batch["future_data"]
        condition = batch["condition"]
        loss = self._get_loss(x, condition)
        return loss

    @torch.no_grad
    def validation_step(self, batch):
        x = batch["future_data"]
        condition = batch["condition"]
        loss = self._get_loss(x, condition)
        return loss

    @torch.no_grad
    def predict_step(self, batch):
        assert self._sample_ready
        condition = batch["condition"]
        x = self._init_noise(batch["future_data"], condition)

        c = self._encode_condition(condition)
        if c.shape[0] != x.shape[0]:
            c = c.repeat(x.shape[0] // c.shape[0], 1)

        all_x = [x]
        for i in range(len(self.sample_Ts)):
            # i is the index of denoise step, t is the denoise step
            t = self.sample_Ts[i]
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])

            # ! norm here?
            # x_norm, _ = self._normalize(x) if self.norm else (x, None)

            # estimate x_0
            x_hat = self.backbone(x, t_tensor, c)
            if self.pred_diff:
                x_hat = x_hat + x

            x_hat_norm, _ = self._normalize(x_hat) if self.norm else (x_hat, None)

            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            x = self.reverse(x, x_hat_norm, t, prev_t)

            if self.norm:
                _, (mu, std) = self._normalize(x_hat, t=prev_t)
            else:
                mu, std = 0.0, 1.0

            if self.collect_all:
                all_x.append(x * std + mu)

        if self.collect_all:
            all_x[0] = all_x[0] + mu
            out_x = torch.stack(all_x, dim=-1)
        else:
            out_x = x + mu
        out_x = out_x.reshape(self.n_sample, *batch["future_data"].shape).detach().cpu()
        return out_x

    def _get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t
        t = torch.randint(0, self.T, (batch_size,)).to(x.device)

        # x^\prime if norm
        x_norm, _ = self._normalize(x) if self.norm else (x, None)

        # corrupt data
        x_noisy = self.degrade(x_norm, t)

        # eps_theta
        c = self._encode_condition(condition)
        x_hat = self.backbone(x_noisy, t, c)
        if self.pred_diff:
            x_hat = x_hat + x_noisy

        # fit on original scale
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def _init_noise(
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
            noise = self.degrade(
                x,
                torch.ones((x.shape[0],), device=x.device, dtype=torch.int)
                * (self.T - 1),
            )

        return noise

    def _encode_condition(self, condition: dict = None):
        if (condition is not None) and (self.conditioner is not None):
            c = self.conditioner(**condition)
        else:
            c = None
        return c

    def config_sampling(
        self,
        n_sample: int = 1,
        sigmas: torch.Tensor = None,
        sample_steps=None,
        collect_all=False,
    ):
        self.n_sample = n_sample
        self.collect_all = collect_all
        self.sample_Ts = (
            list(range(self.T - 1, -1, -1)) if sample_steps is None else sample_steps
        )
        self.sigmas = torch.zeros_like(self.betas) if sigmas is None else sigmas

        for i in range(len(self.sample_Ts) - 1):
            t = self.sample_Ts[i]
            prev_t = self.sample_Ts[i + 1]

            assert self.betas[prev_t] >= self.sigmas[t]
        self._sample_ready = True

    def _normalize(self, x, t=None):
        x_ = self.degrade_fn[t](x) if t is not None else x
        mean = torch.mean(x_, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-6)
        x_norm = (x_ - mean) / stdev
        return x_norm, (mean, stdev)

    def reverse(self, x, x_hat, t, prev_t):
        if t == 0:
            x = x_hat + torch.randn_like(x_hat) * self.sigmas[t]
            # prev_t = None
        else:
            # calculate coeffs
            # prev_t = self.sample_Ts[i + 1]
            coeff = self.betas[prev_t] ** 2 - self.sigmas[t] ** 2
            coeff = torch.sqrt(coeff) / self.betas[t]
            sigma = self.sigmas[t]
            # coeff = 0
            # sigma = self.betas[prev_t]
            x = (
                x * coeff
                + self.degrade_fn[prev_t](x_hat)
                - self.degrade_fn[t](x_hat) * coeff
            )
            x = x + torch.randn_like(x) * sigma
        return x


class MADFreq(MADTime):
    def __init__(
        self,
        backbone_config: Dict,
        conditioner_config: Dict,
        noise_kw={"name": "linear", "min_beta": 0, "max_beta": 0},
        norm=True,
        pred_diff=False,
    ) -> None:
        super().__init__(backbone_config, conditioner_config, noise_kw, norm, pred_diff)
        freq = torch.fft.rfftfreq(self.seq_length)
        self.degrade_fn = [MovingAvgFreq(f, freq=freq) for f in self.factors]
        freq_response = torch.concat([df.Hw for df in self.degrade_fn])
        self.register_buffer("freq_response", freq_response)
        self.betas = self.betas * 1 / 2**0.5

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        x_complex = real_imag_to_complex_freq(x)
        # x_complex = dft(x, real_imag=False)
        # # real+imag --> complex tensor
        # x_complex = real_imag_to_complex_freq(x) if self.freq_kw["real_imag"] else x

        # matrix multiplication
        x_filtered = x_complex * self.freq_response[t]

        # add noise if needed
        x_filtered = x_filtered + self.betas[t].unsqueeze(1).unsqueeze(
            1
        ) * torch.randn_like(x_filtered)

        # complex tensor --> real+imag
        x_noisy = complex_freq_to_real_imag(x_filtered)
        return x_noisy

    def _normalize(self, x, t=None):
        x_ = self.degrade_fn[t](x) if t is not None else x
        mean = torch.zeros_like(x_)
        mean[:, [0], :] = x_[:, [0], :]
        var = torch.sum(torch.abs(x_[1:]) ** 2) * 2 / x.shape[1]
        stdev = torch.sqrt(var + 1e-6)
        x_norm = (x_ - mean) / stdev
        return x_norm, (mean, stdev)


# class MovingAvgDiffusion(BaseDiffusion):
#     """MovingAvgDiffusion. Limit to the average terms be the factors of the seq_length.

#     Following Cold Diffusion

#     Args:
#         BaseDiffusion (_type_): _description_
#     """

#     def __init__(
#         self,
#         seq_length: int,
#         backbone: nn.Module,
#         conditioner: nn.Module = None,
#         freq_kw={"frequency": False},
#         device="cpu",
#         noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
#         only_factor_step=False,
#         norm=True,
#         fit_on_diff=False,
#     ) -> None:
#         self.backbone = backbone.to(device)
#         self.conditioner = conditioner
#         if conditioner is not None:
#             self.conditioner = self.conditioner.to(device)
#             print("CONDITIONAL DIFFUSION!")
#         self.device = device
#         self.freq_kw = freq_kw
#         self.seq_length = seq_length

#         # get int factors
#         middle_res = get_factors(seq_length) + [seq_length]

#         # get 2,3,..., seq_length
#         factors = (
#             middle_res if only_factor_step else [i for i in range(2, seq_length + 1)]
#         )
#         self.T = len(factors)

#         if freq_kw["frequency"]:
#             print("DIFFUSION IN FREQUENCY DOMAIN!")
#             freq = torch.fft.rfftfreq(seq_length)
#             self.degrade_fn = [MovingAvgFreq(f, freq=freq) for f in factors]
#             self.freq_response = torch.concat(
#                 [df.Hw.to(self.device) for df in self.degrade_fn]
#             )
#         else:
#             print("DIFFUSION IN TIME DOMAIN!")
#             self.degrade_fn = [MovingAvgTime(f) for f in factors]

#         noise_schedule = getattr(schedule, noise_kw.pop("name") + "_schedule")(
#             n_steps=self.T, device=device, **noise_kw
#         )
#         # for corruption
#         self.betas = noise_schedule[-1]
#         self.alphas = noise_schedule[-2]
#         if torch.allclose(self.betas, torch.zeros_like(self.betas)):
#             print("COLD DIFFUSION")
#             self.cold = True
#         else:
#             print("HOT DIFFUSION")
#             self.cold = False

#         self.norm = norm
#         self.fit_on_diff = fit_on_diff

#     @torch.no_grad
#     def degrade(self, x: torch.Tensor, t: torch.Tensor):
#         if self.freq_kw["frequency"]:
#             x_complex = dft(x, real_imag=False)
#             # # real+imag --> complex tensor
#             # x_complex = real_imag_to_complex_freq(x) if self.freq_kw["real_imag"] else x

#             # matrix multiplication
#             x_filtered = x_complex * self.freq_response[t]

#             # add noise if needed
#             x_filtered = x_filtered + self.betas[t].unsqueeze(1).unsqueeze(
#                 1
#             ) * torch.randn_like(x_filtered).to(self.device)

#             # complex tensor --> real+imag
#             x_noisy = (
#                 complex_freq_to_real_imag(x_filtered)
#                 if self.freq_kw["real_imag"]
#                 else x_filtered
#             )
#         else:
#             # loop for each sample to temporal average
#             x_noisy = []
#             for i in range(x.shape[0]):
#                 x_n = self.degrade_fn[t[i]](x[[i], ...])
#                 x_noisy.append(x_n)
#             x_noisy = torch.concat(x_noisy)
#             x_noisy = x_noisy + self.betas[t].unsqueeze(1).unsqueeze(
#                 1
#             ) * torch.randn_like(x_noisy).to(self.device)

#         return x_noisy

#     @torch.no_grad
#     def sample(
#         self,
#         x: torch.Tensor,
#         condition: Dict[str, torch.Tensor] = None,
#         collect_all: bool = False,
#     ):
#         c = self._encode_condition(condition)
#         if c.shape[0] != x.shape[0]:
#             print("extending")
#             c = c.repeat(x.shape[0] // c.shape[0], 1)

#         # get denoise steps (DDIM/full)
#         iter_T = self.sample_Ts
#         # iter_T = self.sample_Ts if self.fast_sample else list(range(self.T - 1, -1, -1))

#         all_x = [x]
#         for i in range(len(iter_T)):
#             # i is the index of denoise step, t is the denoise step
#             t = iter_T[i]
#             t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])

#             x_norm, _ = self._normalize(x)

#             if self.freq_kw["frequency"]:
#                 x_norm = dft(x_norm, **self.freq_kw)

#             # estimate x_0
#             x_hat = self.backbone(x_norm, t_tensor, c)
#             if self.fit_on_diff:
#                 x_hat = x_hat + x_norm

#             x_hat_norm, _ = self._normalize(x_hat, freq=self.freq_kw["frequency"])

#             # in frequency domain, it's easier to do iteration in complex tensor
#             if self.freq_kw["frequency"]:
#                 x = dft(x)
#                 # x = real_imag_to_complex_freq(x) if self.freq_kw["real_imag"] else x
#                 x_hat_norm = (
#                     real_imag_to_complex_freq(x_hat_norm)
#                     if self.freq_kw["real_imag"]
#                     else x_hat_norm
#                 )

#             if t == 0:
#                 x = x_hat_norm + torch.randn_like(x_hat_norm) * self.sigmas[t]
#                 prev_t = None
#             else:
#                 # calculate coeffs
#                 prev_t = iter_T[i + 1]
#                 coeff = self.betas[prev_t] ** 2 - self.sigmas[t] ** 2
#                 coeff = torch.sqrt(coeff) / self.betas[t]
#                 sigma = self.sigmas[t]
#                 # coeff = 0
#                 # sigma = self.betas[prev_t]
#                 x = (
#                     x * coeff
#                     + self.degrade_fn[prev_t](x_hat_norm)
#                     - self.degrade_fn[t](x_hat_norm) * coeff
#                 )

#                 x = x + torch.randn_like(x) * sigma

#             # in frequency domain, backbone use real+imag to train
#             if self.freq_kw["frequency"]:
#                 x = idft(x, real_imag=False)
#                 # x_hat = idft(x_hat, **self.freq_kw)
#             if collect_all:
#                 _, (mu, std) = self._normalize(
#                     x_hat, freq=self.freq_kw["frequency"], t=prev_t
#                 )
#                 all_x.append(x * std + mu)

#         if collect_all:
#             all_x[0] = all_x[0] + mu
#             all_x = torch.stack(all_x, dim=-1)
#             # x_ts = [idft(x_t) for x_t in x_ts]
#         return x * std + mu if not collect_all else all_x
#         # return torch.stack(x_ts, dim=-1)

#     def get_loss(self, x, condition: dict = None):
#         batch_size = x.shape[0]
#         # sample t, x_T
#         t = torch.randint(0, self.T, (batch_size,)).to(self.device)

#         x_norm, _ = self._normalize(x)

#         # corrupt data
#         x_noisy = self.degrade(x_norm, t)

#         # eps_theta
#         # GET mu and std in time domain
#         c = self._encode_condition(condition)
#         x_hat = self.backbone(x_noisy, t, c)
#         if self.fit_on_diff:
#             x_hat = x_hat + x_noisy

#         # compute loss
#         if self.freq_kw["frequency"]:
#             x = dft(x, **self.freq_kw)

#         loss = nn.functional.mse_loss(x_hat, x)
#         return loss

#     def init_noise(
#         self,
#         x: torch.Tensor,
#         condition: Dict[str, torch.Tensor] = None,
#         n_sample: int = 1,
#     ):
#         # * INIT ON TIME DOMAIN AND DO TRANSFORM
#         # TODO: condition on basic predicion f(x)
#         if (condition is not None) and (self.conditioner is not None):
#             # ! just for test !
#             # ! use latest observed as zero frequency component !
#             time_value = condition["observed_data"][:, [-1], :]
#             prev_value = torch.zeros_like(time_value) if self.norm else time_value
#             prev_value = prev_value.expand(-1, x.shape[1], -1)

#             noise = (
#                 prev_value
#                 + torch.randn(
#                     n_sample,
#                     *prev_value.shape,
#                     device=prev_value.device,
#                     dtype=prev_value.dtype,
#                 )
#                 * self.betas[-1]
#             )

#         else:
#             # condition on given target
#             noise = self.degrade(
#                 x,
#                 torch.ones((x.shape[0],), device=x.device, dtype=torch.int)
#                 * (self.T - 1),
#             )

#         return noise

#     def _encode_condition(self, condition: dict = None):
#         if (condition is not None) and (self.conditioner is not None):
#             c = self.conditioner(**condition)
#         else:
#             c = None
#         return c

#     def get_params(self):
#         params = list(self.backbone.parameters())
#         if self.conditioner is not None:
#             params += list(self.conditioner.parameters())
#         return params

#     def config_sample(self, sigmas, sample_steps=None):
#         self.sample_Ts = (
#             list(range(self.T - 1, -1, -1)) if sample_steps is None else sample_steps
#         )
#         self.sigmas = sigmas

#         for i in range(len(self.sample_Ts) - 1):
#             t = self.sample_Ts[i]
#             prev_t = self.sample_Ts[i + 1]

#             assert self.betas[prev_t] >= self.sigmas[t]

#     def _normalize(self, x, freq=False, t=None):
#         if t is not None:
#             if freq:
#                 x_ = self.degrade_fn[t](real_imag_to_complex_freq(x))
#                 x_ = complex_freq_to_real_imag(x_)
#             else:
#                 x_ = self.degrade_fn[t](x)

#             if freq:
#                 x_time = idft(x_, **self.freq_kw)
#             else:
#                 x_time = x_

#             mean = torch.mean(x_time, dim=1, keepdim=True)
#             stdev = torch.sqrt(
#                 torch.var(x_time, dim=1, keepdim=True, unbiased=False) + 1e-5
#             )

#             # x_norm = (x_time - mean) / stdev
#             return None, (mean, stdev)
#         else:
#             if freq:
#                 x_time = idft(x, **self.freq_kw)
#             else:
#                 x_time = x

#             mean = torch.mean(x_time, dim=1, keepdim=True)
#             stdev = torch.sqrt(
#                 torch.var(x_time, dim=1, keepdim=True, unbiased=False) + 1e-5
#             )

#             x_norm = (x_time - mean) / stdev

#             if freq:
#                 x_norm = dft(x_norm, **self.freq_kw)
#             return x_norm, (mean, stdev)
