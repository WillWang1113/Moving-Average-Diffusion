import abc
from typing import Dict

import lightning as L
import torch
from torch import nn

from src.models.backbone import build_backbone
from src.models.conditioner import build_conditioner

from ..utils.schedule import noise_schedule
from ..utils.filters import MovingAvgFreq, MovingAvgTime, get_factors
from ..utils.fourier import (
    complex_freq_to_real_imag,
    dft,
    idft,
    real_imag_to_complex_freq,
)


class BaseDiffusion(abc.ABC, L.LightningModule):
    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @torch.no_grad
    def sample(self, batch: dict):
        raise NotImplementedError()

    def _encode_condition(self, condition: Dict[str, torch.Tensor] = None):
        raise NotImplementedError()

    def init_noise(self, batch: dict):
        raise NotImplementedError()


class DDPM(BaseDiffusion):
    def __init__(
        self,
        backbone: nn.Module,
        conditioner: nn.Module = None,
        freq_kw={"frequency": False},
        T=100,
        noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
        device="cpu",
        **kwargs,
    ) -> None:
        noise_kw["n_steps"] = self.T
        alphas, betas, alpha_bars, _ = noise_schedule(noise_kw)
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

    def degrade(self, x, t, noise):
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy

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

    def training_step(self, batch, batch_idx):
        x = batch["future_data"]
        condition = batch["conditions"]

        batch_size = x.shape[0]

        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(x)
        noise = torch.randn_like(x).to(x)

        # corrupt data
        x_noisy = self.degrade(x, t, noise)

        # eps_theta
        c = self._encode_condition(condition)
        eps_theta = self.backbone(x_noisy, t, c)

        # compute loss
        loss = nn.functional.mse_loss(eps_theta, noise)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def _encode_condition(self, condition: dict = None):
        if (condition is not None) and (self.conditioner is not None):
            c = self.conditioner(**condition)
        else:
            c = None
        return c

    def init_noise(
        self,
        batch: torch.Tensor,
        n_sample: int = 1,
    ):
        x = batch["future_data"]
        shape = (n_sample, x.shape[0], x.shape[1], x.shape[2])
        return torch.randn(shape).flatten(end_dim=1).to(x)


class MovingAvgDiffusion(BaseDiffusion):
    """MovingAvgDiffusion. Limit to the average terms be the factors of the seq_length.

    Following Cold Diffusion

    Args:
        BaseDiffusion (_type_): _description_
    """

    def __init__(
        self,
        seq_length: int,
        backbone_config: dict,
        conditioner_config: dict,
        # backbone: nn.Module,
        # conditioner: nn.Module = None,
        freq_kw={"frequency": False},
        # device="cpu",
        noise_kw={"name": "linear", "min_beta": 0.0, "max_beta": 0.0},
        only_factor_step=False,
        norm=True,
        fit_on_diff=False,
        mode="VE",
        lr=2e-4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_config)
        self.conditioner = build_conditioner(conditioner_config)
        self.freq_kw = freq_kw
        self.seq_length = seq_length

        self.lr = lr
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
            self.freq_response = torch.concat([df.Hw for df in self.degrade_fn])
        else:
            print("DIFFUSION IN TIME DOMAIN!")
            self.degrade_fn = [MovingAvgTime(f) for f in factors]

        noise_kw["n_steps"] = self.T
        noise_coeffs = noise_schedule(noise_kw)
        # for corruption
        self.betas = noise_coeffs[-1]
        self.alphas = noise_coeffs[-2]
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
        self.save_hyperparameters()

    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        alpha_ = 1.0 if self.mode == "VE" else self.alphas[t].unsqueeze(1).unsqueeze(1)
        self.freq_response = self.freq_response.to(x.device)
        self.betas = self.betas.to(x)
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

    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        x = self.init_noise(batch, self.n_samples)
        condition = batch["conditions"]
        c = self._encode_condition(condition)
        if c.shape[0] != x.shape[0]:
            print("extending")
            c = c.repeat(x.shape[0] // c.shape[0], 1)

        # get denoise steps (DDIM/full)
        self.sample_Ts = self.sample_Ts
        # self.sample_Ts = self.sample_Ts if self.fast_sample else list(range(self.T - 1, -1, -1))

        all_x = [x]
        for i in range(len(self.sample_Ts)):
            # i is the index of denoise step, t is the denoise step
            t = self.sample_Ts[i]
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])

            x_norm, _ = self._normalize(x)

            if self.freq_kw["frequency"]:
                x_norm = dft(x_norm, **self.freq_kw)

            # estimate x_0
            x_hat = self.backbone(x_norm, t_tensor, c)
            if self.fit_on_diff:
                x_hat = x_hat + x_norm

            x_hat, (mu, std) = self._normalize(x_hat, freq=self.freq_kw["frequency"])

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
                x = x_hat + torch.randn_like(x_hat) * self.sigmas[t]
            else:
                # calculate coeffs
                prev_t = self.sample_Ts[i + 1]
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

            # in frequency domain, backbone use real+imag to train
            if self.freq_kw["frequency"]:
                x = idft(x, real_imag=False)
                # x_hat = idft(x_hat, **self.freq_kw)
            if self.collect_all:
                all_x.append(x)
                all_x[i] = all_x[i] * std + mu

        if self.collect_all:
            all_x[-1] = all_x[-1] * std + mu
            out_x = torch.stack(all_x, dim=-1)
            # x_ts = [idft(x_t) for x_t in x_ts]
            return out_x.detach().cpu()
        else:
            out_x = x * std + mu
            return out_x.detach().cpu()
        # return torch.stack(x_ts, dim=-1)

    def training_step(self, batch, batch_idx):
        x = batch["future_data"]
        condition = batch["conditions"]
        loss = self._get_loss(x, condition)
        self.log(
            "train_loss",
            loss,
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["future_data"]
        condition = batch["conditions"]
        loss = self._get_loss(x, condition)
        self.log(
            "val_loss",
            loss,
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _get_loss(self, x, condition):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(x.device)
        x_norm, _ = self._normalize(x)

        # corrupt data
        x_noisy = self.degrade(x_norm, t)

        # eps_theta
        # GET mu and std in time domain
        c = self._encode_condition(condition)
        x_hat = self.backbone(x_noisy, t, c)
        if self.fit_on_diff:
            x_hat = x_hat + x_noisy

        # compute loss
        if self.freq_kw["frequency"]:
            x = dft(x, **self.freq_kw)

        return torch.nn.functional.mse_loss(x_hat, x)

    def init_noise(
        self,
        batch,
        n_sample: int = 1,
    ):
        x = batch["future_data"]
        condition = batch["conditions"]
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
                ).to(x)
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
            c = self.conditioner(**condition)
        else:
            c = None
            # mu, std = None, None
        return c

    def configure_sampling(
        self, sigmas=None, sample_steps=None, n_samples=100, collect_all=False
    ):
        self.sample_Ts = (
            list(range(self.T - 1, -1, -1)) if sample_steps is None else sample_steps
        )
        self.sigmas = torch.zeros_like(self.betas) if sigmas is None else sigmas
        self.sigmas = self.sigmas.to(self.betas)
        self.n_samples = n_samples
        self.collect_all = collect_all

        for i in range(len(self.sample_Ts) - 1):
            t = self.sample_Ts[i]
            prev_t = self.sample_Ts[i + 1]

            assert self.betas[prev_t] >= self.sigmas[t]

    def _normalize(self, x, freq=False):
        if freq:
            x_time = idft(x, **self.freq_kw)
        else:
            x_time = x

        if self.norm:
            mean = torch.mean(x_time, dim=1, keepdim=True)
            stdev = torch.sqrt(
                torch.var(x_time, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
        else:
            mean, stdev = (
                torch.zeros((x_time.shape[0], 1, x_time.shape[-1])).to(x_time),
                torch.ones((x_time.shape[0], 1, x_time.shape[-1])).to(x_time),
            )
        x_norm = (x_time - mean) / stdev

        if freq:
            x_norm = dft(x_norm, **self.freq_kw)
        return x_norm, (mean, stdev)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
