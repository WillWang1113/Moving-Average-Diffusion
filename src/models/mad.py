from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from ..models.backbone import build_backbone
from ..models.conditioner import build_conditioner
from ..utils.filters import MovingAvgFreq, MovingAvgTime, get_factors


def complex_mse_loss(output, target):
    output_re_im = torch.stack([output.real, output.imag])
    target_re_im = torch.stack([target.real, target.imag])
    return F.mse_loss(output_re_im, target_re_im)


class MAD(L.LightningModule):
    """MovingAvg Diffusion."""

    def __init__(
        self,
        backbone_config: dict,
        conditioner_config: dict,
        noise_schedule: dict,
        norm=True,
        pred_diff=False,
        lr=2e-4,
        alpha=1e-5,
        p_uncond=0.0,
        freq=False,
        factor_only=False,
        stride_equal_to_kernel_size=False,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_config)
        self.conditioner = build_conditioner(conditioner_config)
        self.seq_length = self.backbone.seq_length
        self.norm = norm
        self.pred_diff = pred_diff
        self.lr = lr
        self.alpha = alpha
        self.freq = freq
        self.p_uncond = p_uncond
        self.loss_fn = complex_mse_loss if freq else F.mse_loss

        # config diffusion steps
        self.factors = (
            get_factors(self.seq_length)
            if factor_only
            else list(range(2, self.seq_length + 1))
        )
        if self.freq:
            assert not stride_equal_to_kernel_size
            degrade_class = MovingAvgFreq
        else:
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
        x = batch.pop("future_data")
        # if norm
        x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))
        # if frequency domain
        x = torch.fft.rfft(x, dim=1, norm="ortho") if self.freq else x

        loss = self._get_loss(x, batch, x_mean, x_std)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.pop("future_data")
        # if norm
        x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        # if frequency domain
        x = torch.fft.rfft(x, dim=1, norm="ortho") if self.freq else x

        loss = self._get_loss(x, batch, x_mean, x_std)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        assert self._sample_ready
        x_real = batch.pop("future_data")
        x = self._init_noise(x_real, batch)

        # treat different samples as batch
        x = x.flatten(end_dim=1)

        # fig, ax = plt.subplots()
        # ax.plot(x[0].flatten().cpu(), label='in')

        # DFT, optional
        if self.freq:
            x = torch.fft.rfft(x, dim=1, norm="ortho")

        c = self._encode_condition(batch)

        # align shape, repeat conditions
        for k in c:
            c[k] = c[k].repeat(
                self.n_sample,
                *[1 for _ in range(c[k].ndim - 1)],
            )

        x = self._sample_loop(x, c)

        # IDFT, optional
        if self.freq:
            x = torch.fft.irfft(x, dim=1, norm="ortho")
        # ax.plot(x[0].flatten().cpu(), label='out')

        # denorm
        # mu = c["mean_pred"] if self.norm else 0.0
        # std = c["std_pred"] if self.norm else 1.0
        if self.norm:
            if self.init_model is not None:
                mu = self.init_model_mean.repeat(
                    self.n_sample,
                    *[1 for _ in range(self.init_model_mean.ndim - 1)],
                )
                std = self.init_model_std.repeat(
                    self.n_sample,
                    *[1 for _ in range(self.init_model_std.ndim - 1)],
                )
            else:
                mu, std = c["mean_pred"], c["std_pred"]
        else:
            mu, std = 0.0, 1.0
        out_x = x * std + mu
        # ax.plot(out_x[0].flatten().cpu(), label='denorm_out')
        # ax.legend()
        # fig.savefig('test.png')

        # fig, ax = plt.subplots(3, 3)
        # ax = ax.flatten()
        # x_plot, (_, _) = self._normalize(x_real)
        # for i in range(len(ax)):
        #     choose = np.random.randint(0, x_real.shape[0])
        #     # ax[i].plot(
        #     #     out_x[:3, choose].squeeze().T.cpu(),
        #     # )
        #     ax[i].plot(
        #         x.reshape(
        #             self.n_sample,
        #             x_real.shape[0],
        #             x.shape[1],
        #             x_real.shape[2],
        #         )[:3, choose]
        #         .squeeze()
        #         .T.cpu()
        #     )
        #     ax[i].plot(x_plot[choose].flatten().cpu())
        #     # ax[i].plot(x_real[choose].flatten().cpu())
        # fig.savefig(f"assets/uncond_norm_{batch_idx}.png")
        # plt.close(fig)

        out_x = out_x.reshape(
            self.n_sample,
            x_real.shape[0],
            x.shape[1],
            x_real.shape[2],
        )

        return out_x

    def _sample_loop(self, x: torch.Tensor, c: Dict):
        for i in range(len(self.sample_Ts)):
            # i is the index of denoise step, t is the denoise step
            t = self.sample_Ts[i]
            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
            # if t == 2:
            #     x_plot,(_,_) = self._normalize(x_real)
            #     loss = F.mse_loss(
            #         x.reshape(self.n_sample, -1, self.seq_length, 1).mean(dim=0), x_plot
            #     )
            #     print(loss)

            # fig, ax = plt.subplots()
            # plot_x = (
            #     x.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0].flatten().cpu()
            # )
            # plot_x = torch.fft.irfft(plot_x, norm="ortho")
            # ax.plot(
            #     plot_x,
            #     label="x_t",
            # )

            # pred x_0, conditional
            x_hat_cond = self.backbone(x, t_tensor, c)
            if self.pred_diff:
                x_hat_cond = x_hat_cond + x
                
            # plot_x_hat_cond = (
            #     x_hat_cond.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0].flatten().cpu()
            # )
            # plot_x_hat_cond = torch.fft.irfft(plot_x_hat_cond, norm="ortho")
            # ax.plot(
            #     plot_x_hat_cond,
            #     label="hat x_cond_0",
            # )

            # ax.plot(
            #     x_hat_cond.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0]
            #     .flatten()
            #     .cpu(),
            #     label="hat x_cond_0",
            # )
            # pred x_0, unconditional
            x_hat_uncond = self.backbone(
                x, t_tensor, {k: torch.zeros_like(c[k]) for k in c}
            )
            if self.pred_diff:
                x_hat_uncond = x_hat_uncond + x

            # ax.plot(
            #     x_hat_uncond.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0]
            #     .flatten()
            #     .cpu(),
            #     label="hat x_uncond_0",
            # )
            
            # plot_x_hat_uncond = (
            #     x_hat_uncond.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0].flatten().cpu()
            # )
            # plot_x_hat_uncond = torch.fft.irfft(plot_x_hat_uncond, norm="ortho")
            # ax.plot(
            #     plot_x_hat_uncond,
            #     label="hat x_uncond_0",
            # )

            # mixing
            x_hat = self.w_cond * x_hat_cond + (1 - self.w_cond) * x_hat_uncond

            # ax.plot(
            #     x_hat.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0]
            #     .flatten()
            #     .cpu(),
            #     label="hat x_0",
            # )

            # x_{t-1}
            x = self.reverse(x=x, x_hat=x_hat, t=t, prev_t=prev_t)
            # ax.plot(
            #     x.reshape(self.n_sample, -1, self.seq_length, 1)[0, 0].flatten().cpu(), label='x_t-1'
            # )
            # ax.legend()
            # fig.savefig(f'assets/test{self.factors[t]}_time.png')

        return x

    def _get_loss(self, x, condition: dict = None, x_mean=None, x_std=None):
        batch_size = x.shape[0]

        # sample t
        t = torch.randint(0, self.T, (batch_size,)).to(x.device)

        # corrupt data
        x_noisy = self.degrade(x, t)

        # eps_theta
        c = self._encode_condition(condition)

        # unconditional mask
        mask = torch.rand((batch_size,)).to(x.device) > self.p_uncond
        for k in c:
            c[k] = c[k] * mask.view(
                -1,
                *[1 for _ in range(c[k].ndim - 1)],
            )

        x_hat = self.backbone(x_noisy, t, c)
        if self.pred_diff:
            x_hat = x_hat + x_noisy

        # diffusion loss
        loss = self.loss_fn(x_hat, x)
        # print('original loss')
        # print(loss)

        # regularization
        if x_mean is not None:
            loss += F.mse_loss(c["mean_pred"], x_mean)
        # print('add mean loss')
        # print(loss)

        # TODO: scale too small
        if x_std is not None:
            # optimizing log std to stablize
            # loss += F.mse_loss(c["std_pred"], x_std)
            loss += F.mse_loss(c["std_pred"], x_std)
        # print('add std loss')
        # print(loss)

        return loss

    def _init_noise(
        self,
        x: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
    ):
        # conditional
        if (condition is not None) and (self.conditioner is not None):
            device = condition["observed_data"].device

            first_step = self.sample_Ts[0]
            # print(first_step)
            if self.init_model is None:
                if self.norm:
                    # sample from zero-mean
                    noise = torch.zeros_like(x)

                    # ! TEST:
                else:
                    # assume stationary
                    noise = torch.mean(condition["observed_data"], dim=1, keepdim=True)
                    noise = noise.expand(-1, self.seq_length, -1)
                assert noise.shape == x.shape
            else:
                # t = torch.ones((x.shape[0],), device=x.device, dtype=torch.int) * first_step
                # noise = self.K[t] @ x
                # noise, _ = self._normalize(noise)
                noise = self.init_model(**condition)
                if noise.shape[1] != self.seq_length:
                    noise = F.interpolate(
                        noise.permute(0, 2, 1),
                        size=self.seq_length,
                        mode="linear",
                        # mode="nearest-exact",
                    ).permute(0, 2, 1)
                    # fig, ax = plt.subplots()
                    # ax.plot(noise[0].flatten().cpu())
                    # ax.plot(x[0].flatten().cpu())
                    # fig.savefig('test.png')
                if self.norm:
                    noise, (init_model_mean, init_model_std) = self._normalize(noise)
                    self.init_model_mean = init_model_mean
                    self.init_model_std = init_model_std
                    # print(x.mean(dim=1, keepdim=True)[:10,0,0])
                    # print(init_model_mean[:10,0,0])
                    # print(x.std(dim=1, keepdim=True, unbiased=False)[:10,0,0])
                    # print(init_model_std[:10,0,0])
                assert noise.shape == x.shape

            # fig, ax = plt.subplots()
            # x_plot, (_,_) = self._normalize(x)
            # print(F.mse_loss(noise, x_plot))
            # ax.plot(x_plot[0].flatten().cpu())
            # ax.plot(noise[0].flatten().cpu())
            add_noise = torch.randn(
                self.n_sample,
                *noise.shape,
                device=device,
                dtype=noise.dtype,
            )
            if self.freq:
                add_noise = torch.fft.rfft(add_noise, dim=2, norm="ortho")
                add_noise = add_noise * self.betas[[first_step]].unsqueeze(0).unsqueeze(
                    -1
                )
                add_noise = torch.fft.irfft(add_noise, dim=2, norm="ortho")
            else:
                add_noise = add_noise * self.betas[first_step]
            noise = noise + add_noise
            # ax.plot(noise[0, 0].flatten().cpu())
            # fig.savefig("test.png")
        # TODO: unconditional
        else:
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
        w_cond: float = 1,
        sigmas: torch.Tensor = None,
        sample_steps=None,
        init_model: torch.nn.Module = None,
    ):
        self.init_model = init_model
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
