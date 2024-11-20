import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import src.backbone

from ..utils.filters import MovingAvgTime, get_factors


class MADF(L.LightningModule):
    """MovingAvg Diffusion in Time domain"""

    def __init__(
        self,
        backbone_config: dict,
        noise_schedule: dict,
        norm=True,
        pred_diff=False,
        lr=2e-4,
        alpha=1e-5,
        factor_only=False,
        stride_equal_to_kernel_size=False,
    ) -> None:
        super().__init__()
        bb_class = getattr(src.backbone, backbone_config["name"])
        self.backbone = bb_class(**backbone_config)
        self.seq_length = self.backbone.seq_length
        self.norm = norm
        self.pred_diff = pred_diff
        self.lr = lr
        self.alpha = alpha
        self.loss_fn = F.mse_loss

        # config diffusion steps
        self.factors = (
            get_factors(self.seq_length)
            if factor_only
            else list(range(2, self.seq_length + 1))
        )

        degrade_class = (
            MovingAvgTime
            if not stride_equal_to_kernel_size
            else lambda f, sl: MovingAvgTime(f, sl, stride=f)
        )

        self.T = len(self.factors)
        self.degrade_fn = [degrade_class(f, self.seq_length) for f in self.factors]
        K = torch.stack([df.K for df in self.degrade_fn])

        self.register_buffer("K", K)
        self.register_buffer("betas", noise_schedule["betas"].reshape(self.T, -1))
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.alpha)

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        x_filtered = self.K[t] @ x
        # add noise
        x_noisy = x_filtered + self.betas[t].unsqueeze(-1) * torch.randn_like(
            x_filtered
        )
        return x_noisy

    def training_step(self, batch, batch_idx):
        x = batch.pop("x")
        # if norm
        x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        loss, mean_loss, std_loss = self._get_loss(x, batch, x_mean, x_std)

        log_dict = {
            "recon_loss": loss,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "loss": loss + mean_loss + std_loss,
        }

        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return log_dict

    def validation_step(self, batch, batch_idx):
        x = batch.pop("x")
        # if norm
        x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        # loss = self._get_loss(x, batch, x_mean, x_std)
        loss, mean_loss, std_loss = self._get_loss(x, batch, x_mean, x_std)

        log_dict = {
            "recon_loss": loss,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "val_loss": loss + mean_loss + std_loss,
        }

        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return log_dict

    def predict_step(self, batch, batch_idx):
        assert self._sample_ready

        x_real = batch.pop("x")
        cond = batch.get("c", None)
        all_sample_x = []
        for i in range(self.n_sample):
            x = self._init_noise(x_real, cond)
            x, mu, std = self._sample_loop(x, cond)

            # denorm
            if (cond is None) or (self.condition == "sr"):
                mu, std = self.init_mean, self.init_std
            else:
                if not self.norm:
                    mu, std = 0, 1

            out_x = x * std + mu
            assert out_x.shape == x_real.shape
            all_sample_x.append(out_x)

        all_sample_x = torch.stack(all_sample_x)

        return all_sample_x

    def _sample_loop(self, x: torch.Tensor, c: torch.Tensor):
        for i in range(len(self.sample_Ts)):
            # i is the index of denoise step, t is the denoise step
            t = self.sample_Ts[i]
            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])

            # unconditional inference: super-res or null cond
            if (c is None) or (self.condition == "sr"):
                x_hat, x_mean, x_std = self.backbone(x, t_tensor, None, train=False)
                if self.pred_diff:
                    x_hat = x_hat + x

            # conditional inference: fcst
            else:
                # Classifier-free mixing
                x_concat = torch.concat([x, x], dim=0)
                c_null = torch.zeros_like(c)
                c_concat = torch.concat([c, c_null], dim=0)
                t_concat = torch.concat([t_tensor, t_tensor], dim=0)

                x_hat, x_mean, x_std = self.backbone(
                    x_concat, t_concat, c_concat, train=False
                )

                if self.pred_diff:
                    x_hat = x_hat + x_concat

                cond_x_hat, uncond_x_hat = torch.split(x_hat, len(x_hat) // 2, dim=0)
                cond_x_mean, uncond_x_mean = torch.split(x_mean, len(x_hat) // 2, dim=0)
                cond_x_std, uncond_x_std = torch.split(x_std, len(x_hat) // 2, dim=0)

                # mixing
                x_hat = self.w_cond * cond_x_hat + (1 - self.w_cond) * uncond_x_hat
                x_mean = self.w_cond * cond_x_mean + (1 - self.w_cond) * uncond_x_mean
                x_std = self.w_cond * cond_x_std + (1 - self.w_cond) * uncond_x_std

            # x_{t-1}
            x = self.reverse(x=x, x_hat=x_hat, t=t, prev_t=prev_t)

        return x, x_mean, x_std

    def _get_loss(self, x, condition: dict = None, x_mean=None, x_std=None):
        batch_size = x.shape[0]
        cond = condition.get("c", None)
        # cond = condition["c"]

        # sample t
        t = torch.randint(0, self.T, (batch_size,)).to(x.device)

        # corrupt data
        x_noisy = self.degrade(x, t)

        x_hat, x_mean_hat, x_std_hat = self.backbone(x_noisy, t, cond, train=True)
        if self.pred_diff:
            x_hat = x_hat + x_noisy

        # diffusion loss
        loss = self.loss_fn(x_hat, x)

        # regularization
        if (x_mean is not None) and (cond is not None):
            mean_loss = F.mse_loss(x_mean_hat, x_mean)
        else:
            mean_loss = 0

        # TODO: scale too small?
        if (x_std is not None) and (cond is not None):
            # optimizing log std to stablize
            std_loss = F.mse_loss(x_std_hat, x_std)
        else:
            std_loss = 0

        return loss, mean_loss, std_loss

    def _init_noise(
        self,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ):
        first_step = self.sample_Ts[0]

        # unconditional generation: start from p( degrade_fn(x_0) )
        if self.condition is None:
            assert condition is None

            if self.norm:
                x_norm, (self.init_mean, self.init_std) = self._normalize(x)
            else:
                x_norm, self.init_mean, self.init_std = x, 0, 1

            t = torch.ones((x.shape[0],), device=x.device, dtype=torch.int) * first_step
            x_T = self.degrade(x_norm, t)

        # unconditional generation: super-res, start from given p(x_t)
        elif self.condition == "sr":
            assert condition is not None

            if self.norm:
                if self.norm:
                    x_norm, (self.init_mean, self.init_std) = self._normalize(condition)
                else:
                    x_norm, self.init_mean, self.init_std = condition, 0, 1

            t = torch.ones((x.shape[0],), device=x.device, dtype=torch.int) * first_step
            add_noise = self.betas[t].unsqueeze(-1) * torch.randn_like(condition)
            x_T = condition + add_noise

        # conditional generation: forecast, start from given p(x_T)
        elif self.condition == "fcst":
            assert condition is not None

            if self.norm:
                # sample from zero-mean
                x_T = torch.zeros_like(x)
            else:
                # assume stationary
                x_T = torch.mean(condition, dim=1, keepdim=True)
                x_T = x_T.expand(-1, self.seq_length, -1)

            assert x_T.shape == x.shape
            add_noise = torch.randn_like(x_T)
            add_noise = add_noise * self.betas[first_step]
            x_T = x_T + add_noise

        return x_T

    def config_sampling(
        self,
        n_sample: int = 1,
        w_cond: float = 1,
        sigmas: torch.Tensor = None,
        sample_steps=None,
        condition=None,
    ):
        assert condition in ["fcst", "sr", None]
        self.condition = condition

        self.n_sample = n_sample
        self.w_cond = w_cond
        self.sample_Ts = list(range(self.T)) if sample_steps is None else sample_steps
        self.sample_Ts.sort(reverse=True)
        sigmas = torch.zeros_like(self.betas) if sigmas is None else sigmas
        self.register_buffer("sigmas", sigmas)

        # check
        if sigmas is not None:
            for i in range(len(self.sample_Ts)):
                t = self.sample_Ts[i]
                prev_t = None if t == 0 else self.sample_Ts[i + 1]
                if t != 0:
                    assert t > prev_t
                    assert (self.betas[prev_t] >= self.sigmas[t]).all()
        self._sample_ready = True

    def _normalize(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-6)
        x_norm = (x - mean) / stdev
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
