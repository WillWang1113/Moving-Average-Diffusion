import abc
from typing import Dict
import torch
from torch import nn
from ..schedules.collection import linear_schedule
from ..utils.fourier import complex_freq_to_real_imag, idft, real_imag_to_complex_freq
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
        noise_kw={"min_beta": 1e-4, "max_beta": 2e-2},
        # min_beta=1e-4,
        # max_beta=2e-2,
        device="cpu",
        **kwargs,
    ) -> None:
        alphas, betas, alpha_bars = linear_schedule(
            noise_kw["min_beta"], noise_kw["max_beta"], T, device
        )
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
        noise_kw=None,
        fast_sample=False,
        only_factor_step=True,
    ) -> None:
        self.backbone = backbone.to(device)
        self.conditioner = conditioner
        if conditioner is not None:
            self.conditioner = self.conditioner.to(device)
            print("CONDITIONAL DIFFUSION!")
        self.device = device
        self.freq_kw = freq_kw
        self.seq_length = seq_length
        self.fast_sample = fast_sample

        # get int factors
        middle_res = get_factors(seq_length) + [seq_length]

        # get 2,3,..., seq_length
        factors = (
            middle_res if only_factor_step else [i for i in range(2, seq_length + 1)]
        )
        print("moving average terms:")
        print(factors)

        self.T = len(factors)
        self.sample_Ts = [factors.index(i) for i in middle_res]
        self.sample_Ts.reverse()
        if freq_kw["frequency"]:
            print("DIFFUSION IN FREQUENCY DOMAIN!")
            freq = torch.fft.rfftfreq(seq_length)
            self.freq_response = torch.concat(
                [MovingAvgFreq(f, freq=freq).Hw.to(self.device) for f in factors]
            )
            self.degrade_fn = [
                MovingAvgFreq(f, freq=freq, real_imag=freq_kw["real_imag"])
                for f in factors
            ]

        else:
            print("DIFFUSION IN TIME DOMAIN!")
            self.degrade_fn = [MovingAvgTime(f) for f in factors]
        if noise_kw is not None:
            print("HOT DIFFUSION!")
            _, _, alpha_bars = linear_schedule(
                noise_kw["min_beta"], noise_kw["max_beta"], self.T, device
            )
            self.sigmas = torch.sqrt(1 - alpha_bars)
            self.noise = True
        else:
            print("COLD DIFFUSION!")
            self.noise = False

    # # TODO: too slow
    # @torch.no_grad
    # def forward(self, x: torch.Tensor, t: torch.Tensor):
    #     unique_t, indexes = torch.unique(t, return_inverse=True, sorted=True)
    #     unique_idx = torch.unique(
    #         indexes,
    #         sorted=True,
    #     )
    #     x_noisy = torch.empty_like(x, device=x.device)
    #     for i in unique_idx:
    #         sub_t = unique_t[i]
    #         degrade_fn = self.degrade_fn[sub_t]
    #         x_noisy[indexes == i] = degrade_fn(x[indexes == i])
    #         if self.noise:
    #             noise = self.sigmas[sub_t] * torch.randn_like(x[indexes == i]).to(
    #                 self.device
    #             )
    #             x_noisy[indexes == i] = x_noisy[indexes == i] + noise
    #     return x_noisy

    # TODO: too slow
    @torch.no_grad
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if self.freq_kw["frequency"]:
            # real+imag --> complex tensor
            x_complex = real_imag_to_complex_freq(x) if self.freq_kw["real_imag"] else x

            # matrix multiplication
            x_filtered = x_complex * self.freq_response[t]

            # add noise if needed
            if self.noise:
                x_filtered = x_filtered + self.sigmas[t].unsqueeze(1).unsqueeze(
                    1
                ) * torch.randn_like(x_filtered).to(self.device)

            # complex tensor --> real+imag
            x_noisy = (
                complex_freq_to_real_imag(x_filtered, x.shape[1])
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
            if self.noise:
                x_noisy = x_noisy + self.sigmas[t].unsqueeze(1).unsqueeze(
                    1
                ) * torch.randn_like(x_noisy).to(self.device)

        return x_noisy

    @torch.no_grad
    def backward(self, x: torch.Tensor, condition: Dict[str, torch.Tensor] = None):
        c = self._encode_condition(condition)
        if c.shape[0] != x.shape[0]:
            c = c.repeat(x.shape[0] // c.shape[0], 1)

        iter_T = self.sample_Ts if self.fast_sample else range(self.T - 1, -1, -1)

        for t in iter_T:
            t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])
            x_hat = self.backbone(x, t_tensor, c)

            if self.noise:
                # FOR NOISE, DDIM(\eta = 0)
                coeff = (self.sigmas[t - 1]**2 - 0.01**2) / self.sigmas[t]**2
                coeff = torch.sqrt(coeff)
                if t == 0:
                    x = x_hat
                else:
                    x = (
                        x * coeff
                        + self.degrade_fn[t - 1](x_hat)
                        - self.degrade_fn[t](x_hat) * coeff + torch.randn_like(x) * 0.01
                    )
            else:
                # FOR DETERMINISTIC
                if t == 0:
                    x = x - self.degrade_fn[t](x_hat) + x_hat
                else:
                    x = x - self.degrade_fn[t](x_hat) + self.degrade_fn[t - 1](x_hat)

        if self.freq_kw["frequency"]:
            x = idft(x, **self.freq_kw)
            # x_ts = [idft(x_t) for x_t in x_ts]
        return x
        # return torch.stack(x_ts, dim=-1)

    def get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy = self.forward(x, t)

        # look at corrupted data
        # fig, ax = plt.subplots()
        # ax.plot(x_noisy[0].detach().cpu())
        # fig.suptitle(f'{t[0].detach().cpu()}')
        # fig.savefig(f'assets/noisy_{t[0]}.png')
        # plt.close()

        # eps_theta
        c = self._encode_condition(condition)
        x_hat = self.backbone(x_noisy, t, c)

        # compute loss
        if self.freq_kw["frequency"] and (not self.freq_kw["real_imag"]):
            x = torch.fft.irfft(x, dim=1, norm="ortho")
            x_hat = torch.fft.irfft(x_hat, dim=1, norm="ortho")
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def init_noise(
        self,
        x: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
        n_sample: int = 1,
    ):
        # TODO: condition on basic predicion f(x)
        if (condition is not None) and (self.conditioner is not None):
            # ! just for test !
            # ! use observed last !
            if self.freq_kw["frequency"]:
                time_value = condition["observed_data"]

                # ortho norm: factor of sqrt(seq_len)
                prev_value = time_value[:, [-1], :] * torch.sqrt(
                    torch.tensor(time_value.shape[1], device=x.device)
                )
                if self.noise:
                    prev_value = torch.concat(
                        [
                            prev_value,
                            torch.zeros(
                                x.shape[0],
                                time_value.shape[1] // 2,
                                x.shape[2],
                                device=x.device,
                            ),
                        ],
                        dim=1,
                    )
                    prev_value = prev_value.to(torch.cfloat)
                    noise = (
                        prev_value
                        + torch.randn(
                            n_sample,
                            *prev_value.shape,
                            device=x.device,
                            dtype=torch.cfloat,
                        )
                        * self.sigmas[-1]
                    ).flatten(end_dim=1)

                else:
                    noisy_zero_freq = prev_value + torch.randn(
                        n_sample, *prev_value.shape, device=x.device
                    ) * 0.1 * prev_value.unsqueeze(0)
                    noisy_zero_freq = noisy_zero_freq.view(-1, 1, x.shape[2])

                    noise = torch.concat(
                        [
                            noisy_zero_freq,
                            torch.zeros(
                                n_sample * x.shape[0],
                                x.shape[1] // 2,
                                x.shape[2],
                                device=x.device,
                            ),
                        ],
                        dim=1,
                    ).to(torch.cfloat)
                if self.freq_kw["real_imag"]:
                    noise = complex_freq_to_real_imag(noise, orig_seq_length=x.shape[1])

            else:
                if self.noise:
                    prev_value = condition["observed_data"][:, [-1], :].expand(
                        -1, x.shape[1], -1
                    )
                    noise = (
                        prev_value
                        + torch.randn(n_sample, *prev_value.shape, device=x.device)
                        * self.sigmas[-1]
                    ).flatten(end_dim=1)

                else:
                    prev_value = condition["observed_data"][:, [-1], :]
                    prev_value = (
                        prev_value
                        + torch.randn(n_sample, *prev_value.shape, device=x.device)
                        * 0.1
                        * prev_value.unsqueeze(0)
                    ).view(-1, 1, x.shape[2])
                    noise = prev_value.expand(-1, x.shape[1], -1)

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
        return c

    def get_params(self):
        params = list(self.backbone.parameters())
        if self.conditioner is not None:
            params += list(self.conditioner.parameters())
        return params
