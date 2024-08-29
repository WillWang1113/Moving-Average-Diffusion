from typing import Dict

import torch
import lightning as L
from ..models.backbone import build_backbone
from ..models.base import BaseDiffusion
from ..models.conditioner import build_conditioner
from ..utils.filters import MovingAvgFreq, MovingAvgTime


def complex_mse_loss(output, target):
    output_re_im = torch.stack([output.real, output.imag])
    target_re_im = torch.stack([target.real, target.imag])
    return torch.nn.functional.mse_loss(output_re_im, target_re_im)


class MADTime(L.LightningModule):
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
        self.degrade_fn = [MovingAvgTime(f, self.seq_length) for f in self.factors]
        K = torch.stack([df.K for df in self.degrade_fn])
        self.register_buffer("K", K)
        # ns = noise_kw.pop("name")
        # noise_schedule = getattr(schedule, ns + "_schedule")(n_steps=self.T, **noise_kw)
        self.register_buffer("betas", noise_schedule["betas"].reshape(self.T, -1))
        assert len(self.betas) == self.T
        self.lr = lr
        self.alpha = alpha
        self.loss_fn = torch.nn.functional.mse_loss
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
        x = batch.pop("future_data")
        
        loss = self._get_loss(x, batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # for n, p in self.named_parameters():
        #     if n.__contains__('backbone'):
        #         loss += self.alpha * torch.norm(p, p=1)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.pop("future_data")
        loss = self._get_loss(x, batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        assert self._sample_ready
        x_real = batch.pop("future_data")
        x = self._init_noise(x_real, batch)
        c = self._encode_condition(batch)
        x = x.flatten(end_dim=1)

        # ! test
        x, all_x, mu, std = self._sample_loop(x, c)
        # _, mu, std = c
        # x, all_x, mu, std = self._sample_loop(x, c)

        if self.collect_all:
            all_x[0] = all_x[0] + mu
            out_x = torch.stack(all_x, dim=-1)
            out_x = out_x.reshape(
                self.n_sample,
                x_real.shape[0],
                x.shape[1],
                # self.seq_length // 2 + 1,
                x_real.shape[2],
                -1,
            )
            # out_x = out_x.reshape(self.n_sample, *x_real.shape, -1)
        else:
            out_x = x * std + mu
            out_x = out_x.reshape(
                self.n_sample,
                x_real.shape[0],
                x.shape[1],
                # self.seq_length // 2 + 1,
                x_real.shape[2],
            )
            # out_x = out_x.reshape(self.n_sample, *x_real.shape)
        return out_x

    # ! TEST
    def _sample_loop(self, x: torch.Tensor, c: torch.Tensor):
        if isinstance(c, tuple):
            if c[0].shape[0] != x.shape[0]:
                c = tuple(
                    [
                        c[i].repeat(
                            x.shape[0] // c[i].shape[0],
                            *[1 for _ in range(len(c[i].shape) - 1)],
                        )
                        for i in range(len(c))
                    ]
                )
        else:
            if c.shape[0] != x.shape[0]:
                c = c.repeat(
                    x.shape[0] // c.shape[0], *[1 for _ in range(len(c.shape) - 1)]
                )
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



            x = self.reverse(x=x, x_hat=x_hat, t=t, prev_t=prev_t)
            if self.norm:
                # ! TEST Comments
                _, mu, std = c
                if torch.is_complex(x):
                    mu = torch.concat([mu, torch.zeros_like(x[:, 1:, :])], dim=1)
                # _, (mu, std) = self._normalize(x_hat, t=prev_t)
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

        # # fit on original scale
        # loss = self.loss_fn(x_hat, x)

        # ! TEST
        loss = self.loss_fn(x_hat, x_norm)

        if isinstance(c, tuple):
            _, mean_pred, std_pred = c
            if torch.is_complex(x):
                var = (
                    torch.sum(torch.abs(x[:, 1:, :]) ** 2, dim=1, keepdim=True)
                    * 2
                    / self.seq_length
                )
                stdev = torch.sqrt(var + 1e-6)
                assert stdev.shape == std_pred.shape
                loss = (
                    loss
                    + torch.nn.functional.mse_loss(mean_pred, x[:, [0], :].real)
                    / self.seq_length
                    + torch.nn.functional.mse_loss(std_pred, stdev)
                )

            else:
                loss = (
                    loss
                    + torch.nn.functional.mse_loss(
                        mean_pred, x.mean(dim=1, keepdim=True)
                    )
                    + torch.nn.functional.mse_loss(std_pred, x.std(dim=1, keepdim=True))
                )

        return loss

    def _init_noise(
        self,
        x: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
    ):

        # multiple samples
        # * INIT ON TIME DOMAIN AND DO TRANSFORM
        # TODO: condition on basic predicion f(x)
        if (condition is not None) and (self.conditioner is not None):
            # just for test !
            # use latest observed as zero frequency component !
            device = condition["observed_data"].device
            if self.init_model:
                time_value = self.init_model(**condition)
                # For test
                # time_value = torch.mean(time_value, dim=1, keepdim=True)
            else:
                # time_value = self.conditioner(**condition)[-1] / (self.seq_length ** 0.5)
                time_value = torch.mean(condition["observed_data"], dim=1, keepdim=True)
                # time_value = time_value + 0.1*torch.randn(
                #     10, *time_value.shape, device=time_value.device, dtype=time_value.dtype
                # )
                # time_value = time_value.flatten(end_dim=1)
                # time_value = condition["observed_data"][:, [-1], :]
            assert time_value.shape[1] == 1


            x_noisy = torch.zeros_like(time_value) if self.norm else time_value
            x_noisy = x_noisy.expand(-1, self.seq_length, -1)
            noise = x_noisy + torch.randn(
                self.n_sample,
                *x_noisy.shape,
                device=device,
                dtype=x_noisy.dtype,
            )

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
        init_model: torch.nn.Module = None,
        # init_pred: torch.Tensor= None
        # flatten_nosie=True,
    ):
        # self.init_pred = init_pred
        self.init_model = init_model
        self.n_sample = n_sample
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
        self.loss_fn = lambda x, y: complex_mse_loss(x, y, self.seq_length)

    # torch.fft.rfft version
    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        # matrix multiplication
        x_filtered = x * self.freq_response[t]
        # print(x_filtered.shape)

        # add noise
        x_noisy = x_filtered + self.betas[t].unsqueeze(-1) * torch.randn_like(
            x_filtered
        )

        return x_noisy



    def _normalize(self, x, t=None):

        x_ = self.degrade_fn[t](x) if t is not None else x
        mean = torch.zeros_like(x_)
        mean[:, [0], :] = x_[:, [0], :]
        var = (
            torch.sum(torch.abs(x_[:, 1:, :]) ** 2, dim=1, keepdim=True)
            * 2
            / self.seq_length
        )
        stdev = torch.sqrt(var + 1e-6)
        x_norm = (x_ - mean) / stdev
        return x_norm, (mean, stdev)

    def training_step(self, batch, batch_idx):
        batch["future_data"] = torch.fft.rfft(batch["future_data"], dim=1, norm="ortho")
        # batch["future_data"] = dft(batch["future_data"])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        batch["future_data"] = torch.fft.rfft(batch["future_data"], dim=1, norm="ortho")
        # batch["future_data"] = dft(batch["future_data"])
        return super().validation_step(batch, batch_idx)

    def predict_step(self, batch):
        pred = super().predict_step(batch)
        # print(pred.shape)
        final_shape = pred.shape
        pred_idft = torch.fft.irfft(pred, dim=2, norm="ortho")
        # pred_idft = idft(pred.flatten(end_dim=1)).reshape(final_shape)
        return pred_idft

    def _init_noise(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        noise = super()._init_noise(x, condition)
        # out_noise = dft(noise)
        out_noise = torch.fft.rfft(noise, dim=2, norm="ortho")
        return out_noise
