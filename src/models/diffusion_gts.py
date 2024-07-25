from typing import Dict

import torch
from matplotlib import pyplot as plt
from torch import nn
import lightning as L
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.scaler import StdScaler
from .backbone import build_backbone
from .base import BaseDiffusion
from .conditioner import build_conditioner
from ..utils import schedule
from ..utils.filters import MovingAvgFreq, MovingAvgTime, get_factors
from ..utils.fourier import (
    complex_freq_to_real_imag,
    dft,
    idft,
    real_imag_to_complex_freq,
)


class MADTime(BaseDiffusion, L.LightningModule):
    """MovingAvgDiffusionTime."""

    def __init__(
        self,
        backbone_config: dict,
        conditioner_config: dict,
        noise_schedule: dict,
        norm=True,
        pred_diff=False,
        lr=2e-4,
        alpha=1e-3,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_config)
        self.conditioner = build_conditioner(conditioner_config)
        if self.conditioner is not None:
            print("Conditional Diffusion!")
        self.seq_length = self.backbone.seq_length
        self.norm = norm
        self.pred_diff = pred_diff

        # config diffusion steps
        self.factors = [i for i in range(2, self.seq_length + 1)]
        self.T = len(self.factors)
        self.degrade_fn = [MovingAvgTime(f) for f in self.factors]
        # ns = noise_kw.pop("name")
        # noise_schedule = getattr(schedule, ns + "_schedule")(n_steps=self.T, **noise_kw)
        self.register_buffer("betas", noise_schedule["betas"].reshape(self.T, -1))
        assert len(self.betas) == self.T
        self.lr = lr
        self.alpha = alpha
        self.scaler = StdScaler(dim=1, keepdim=True)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        # loop for each sample to temporal average
        x_noisy = []
        for i in range(x.shape[0]):
            x_n = self.degrade_fn[t[i]](x[[i], ...])
            x_noisy.append(x_n)
        x_noisy = torch.concat(x_noisy)
        x_noisy = x_noisy + self.betas[t].unsqueeze(-1) * torch.randn_like(x_noisy)
        return x_noisy

    def training_step(self, batch, batch_idx):
        x = batch.pop("future_target")
        batch_size = x.shape[0]
        batch["past_target"], loc, scale = self.scaler(
            batch["past_target"], weights=torch.ones_like(batch["past_target"])
        )
        x = (x - loc) / scale
        # x = x.unsqueeze(-1)
        loss = self._get_loss(x, batch)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        for n, p in self.named_parameters():
            if n.__contains__("backbone") and n.__contains__("net"):
                loss += self.alpha * torch.norm(p, p=1)
        return loss

    def validation_step(self, batch):
        x = batch.pop("future_target")
        batch_size = x.shape[0]
        batch["past_target"], loc, scale = self.scaler(
            batch["past_target"], weights=torch.ones_like(batch["past_target"])
        )
        x = (x - loc) / scale
        # x = x.unsqueeze(-1)
        loss = self._get_loss(x, batch)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss
    
    def forward(self, past_target, **kwargs: torch.Any) -> torch.Any:
        # assert self._sample_ready
        # x_real = torch.zeros(past_target.shape[0], self.seq_length, past_target.shape[-1]).to(past_target.device)
        # x_real = kwargs.get('future_target', defaults)
        batch = {'past_target':past_target}
        target_shape = (past_target.shape[0], self.seq_length, past_target.shape[2])
        # x_real = x_real.unsqueeze(-1)
        x = self._init_noise(condition=batch)
        c = self._encode_condition(batch)
        # x = x.flatten(end_dim=1)
        x, all_x, mu, std = self._sample_loop(x, c)
        if self.collect_all:
            all_x[0] = all_x[0] + mu
            out_x = torch.stack(all_x, dim=-1)
            out_x = out_x.reshape(*target_shape, -1)
        else:
            out_x = x * std + mu
            out_x = out_x.reshape(*target_shape)
        return out_x

    def predict_step(self, batch):
        assert self._sample_ready
        return self(**batch)
        # x_real = batch.pop("future_target")
        # # x_real = x_real.unsqueeze(-1)
        # x = self._init_noise(x_real, batch)
        # c = self._encode_condition(batch)
        # # x = x.flatten(end_dim=1)
        # x, all_x, mu, std = self._sample_loop(x, c)
        # if self.collect_all:
        #     all_x[0] = all_x[0] + mu
        #     out_x = torch.stack(all_x, dim=-1)
        #     out_x = out_x.reshape(*x_real.shape, -1)
        # else:
        #     out_x = x * std + mu
        #     out_x = out_x.reshape(*x_real.shape)
        # return out_x

    def _sample_loop(self, x: torch.Tensor, c: torch.Tensor):
        all_x = [x]
        for i in range(len(self.sample_Ts)):
            # i is the index of denoise step, t is the denoise step
            t = self.sample_Ts[i]
            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])

            # ! norm here?
            # x_norm, _ = self._normalize(x) if self.norm else (x, None)

            # estimate x_0
            x_hat = self.backbone(x, t_tensor, c)
            if self.pred_diff:
                x_hat = x_hat + x

            x_hat_norm, _ = self._normalize(x_hat) if self.norm else (x_hat, None)
            x = self.reverse(x=x, x_hat=x_hat_norm, t=t, prev_t=prev_t)
            if self.norm:
                _, (mu, std) = self._normalize(x_hat, t=prev_t)
            else:
                mu, std = 0.0, 1.0

            if self.collect_all:
                all_x.append(x * std + mu)
        return x, all_x, mu, std

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
        x: torch.Tensor = None,
        condition: Dict[str, torch.Tensor] = None,
    ):
        # * INIT ON TIME DOMAIN AND DO TRANSFORM
        # TODO: condition on basic predicion f(x)
        if (condition is not None) and (self.conditioner is not None):
            # ! just for test !
            # ! use latest observed as zero frequency component !
            device = condition["past_target"].device
            time_value = condition["past_target"][:, [-1], :]
            # target_shape = (time_value.shape[0], self.seq_length, time_value.shape[-1])
            
            prev_value = (
                torch.zeros(*time_value.shape, device=device, dtype=time_value.dtype)
                if self.norm
                else time_value
            )
            prev_value = prev_value.expand(-1, self.seq_length, -1)
            noise = prev_value + torch.randn_like(prev_value)

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
        # n_sample: int = 1,
        sigmas: torch.Tensor = None,
        sample_steps=None,
        collect_all=False,
        # flatten_nosie=True,
    ):
        # self.n_sample = n_sample
        self.collect_all = collect_all
        self.sample_Ts = (
            list(range(self.T - 1, -1, -1)) if sample_steps is None else sample_steps
        )
        sigmas = torch.zeros_like(self.betas) if sigmas is None else sigmas
        # self.flat_noise = flatten_nosie
        # self.sigmas = self.sigmas.to(self.betas.device)
        self.register_buffer("sigmas", sigmas)
        for i in range(len(self.sample_Ts)):
            t = self.sample_Ts[i]
            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            if t != 0:
                assert t > prev_t
                assert (self.betas[prev_t] >= self.sigmas[t]).all()
        self._sample_ready = True

    def _normalize(self, x, t=None):
        x_ = self.degrade_fn[t](x) if t is not None else x
        mean = torch.mean(x_, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-6)
        x_norm = (x_ - mean) / stdev
        return x_norm, (mean, stdev)

    def reverse(self, x, x_hat, t, prev_t):
        if t == 0:
            x = x_hat + torch.randn_like(x_hat) * self.sigmas[[t]].unsqueeze(-1)
        else:
            # calculate coeffs
            coeff = self.betas[[prev_t]] ** 2 - self.sigmas[[t]] ** 2
            coeff = torch.sqrt(coeff) / self.betas[[t]]
            sigma = self.sigmas[[t]]
            coeff = coeff.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            x = (
                x * coeff
                + self.degrade_fn[prev_t](x_hat)
                - self.degrade_fn[t](x_hat) * coeff
            )

            x = x + torch.randn_like(x) * sigma
        return x

    def get_predictor(self, input_transform, batch_size=128):
        return PyTorchPredictor(
            prediction_length=self.seq_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            # forecast_generator=DistributionForecastGenerator(self.distr_output),
        )


class MADFreq(MADTime):
    def __init__(
        self,
        backbone_config: Dict,
        conditioner_config: Dict,
        noise_schedule: torch.Tensor,
        norm=True,
        pred_diff=False,
        lr=2e-4,
        alpha=1e-3,
    ) -> None:
        super().__init__(
            backbone_config,
            conditioner_config,
            noise_schedule,
            norm,
            pred_diff,
            lr,
            alpha,
        )
        self.degrade_fn = [
            MovingAvgFreq(f, seq_length=self.seq_length) for f in self.factors
        ]
        freq_response = torch.concat([df.Hw for df in self.degrade_fn])
        self.register_buffer("freq_response", freq_response)
        # self.betas = self.betas

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        # print('degrade')
        # print(x.shape)
        x_complex = real_imag_to_complex_freq(x)
        # print(x_complex.shape)
        # x_complex = dft(x, real_imag=False)
        # # real+imag --> complex tensor

        # matrix multiplication
        x_filtered = x_complex * self.freq_response[t]
        # print(x_filtered.shape)

        # IF REAL and IMAG, noise schedule will be different
        if self.betas.shape[1] > x_filtered.shape[1]:
            beta = self.betas[..., : x_filtered.shape[1]]
        else:
            beta = self.betas

        # add noise
        x_filtered = x_filtered + beta[t].unsqueeze(-1) * torch.randn_like(x_filtered)
        # print(x_filtered.shape)

        # complex tensor --> real+imag
        x_noisy = complex_freq_to_real_imag(x_filtered, seq_len=x.shape[1])
        # print(x_noisy.shape)
        return x_noisy

    def _normalize(self, x, t=None):
        x_ = self.degrade_fn[t](x) if t is not None else x
        mean = torch.zeros_like(x_)
        mean[:, [0], :] = x_[:, [0], :]
        var = (
            torch.sum(torch.abs(x_[:, 1:, :]) ** 2, dim=1, keepdim=True)
            * 2
            / x.shape[1]
        )
        stdev = torch.sqrt(var + 1e-6)
        x_norm = (x_ - mean) / stdev
        return x_norm, (mean, stdev)

    def training_step(self, batch, batch_idx):
        batch["future_target"] = dft(batch["future_target"])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        batch["future_target"] = dft(batch["future_target"])
        return super().validation_step(batch)

    def forward(self, past_target, **kwargs: torch.Any) -> torch.Any:
        pred = super().forward(past_target, **kwargs)
        pred_idft = idft(pred)
        return pred_idft

    def _init_noise(self, x: torch.Tensor=None, condition: Dict[str, torch.Tensor] = None):
        noise = super()._init_noise(x, condition)
        out_noise = dft(noise)
        return out_noise
