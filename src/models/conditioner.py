import sys
import torch
from torch import nn
from torchvision.ops import MLP

from src.models.blocks import RevIN
from ..layers.Autoformer_EncDec import series_decomp
from ..utils.fourier import dft

thismodule = sys.modules[__name__]


# TODO: build backbone function
def build_conditioner(cn_config):
    cn_config_c = cn_config.copy()
    cn_net = getattr(thismodule, cn_config_c.pop("name"), None)
    if cn_net is not None:
        return cn_net(**cn_config_c)
    else:
        return None


class MLPConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        target_seq_length,
        target_seq_channels,
        future_seq_channels=None,
        future_seq_length=None,
        # norm=True,
    ) -> None:
        super().__init__()
        all_input_channel = seq_channels * seq_length

        # include external features
        if future_seq_channels is not None and future_seq_length is not None:
            all_input_channel += future_seq_channels * future_seq_length

        self.latent_dim = latent_dim
        # self.encode_tp = encode_tp
        self.input_enc = MLP(
            in_channels=all_input_channel,
            hidden_channels=[hidden_size, hidden_size, latent_dim],
        )

        # self.norm = norm
        self.seq_channels = seq_channels
        # if norm:
        #     self.mu_net = nn.Linear(latent_dim, 1 * seq_channels)
        #     self.std_net = nn.Linear(latent_dim, 1 * seq_channels)

    def forward(self, past_target, future_features=None, **kwargs):
        x = past_target
        trajs_to_encode = x.flatten(1)  # (batch_size, input_ts, input_dim)

        # # TEST: frequency encoding
        # freq_component = torch.fft.rfft(observed_data, dim=1).flatten(1)
        # theta, phi = complex2sphere(freq_component.real, freq_component.imag)
        # trajs_to_encode = torch.concat([theta, phi, trajs_to_encode], dim=-1)

        if future_features is not None:
            ff = future_features
            trajs_to_encode = torch.concat([trajs_to_encode, ff.flatten(1)], axis=-1)
        out = self.input_enc(trajs_to_encode)

        # if self.norm:
        #     mu = self.mu_net(out).reshape((-1, 1, self.seq_channels))
        #     std = self.std_net(out).reshape((-1, 1, self.seq_channels))
        # else:
        #     mu, std = (
        #         torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device),
        #         torch.ones((x.shape[0], 1, x.shape[-1]), device=x.device),
        #     )
        return out


class CIMLPConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        target_seq_length,
        target_seq_channels,
        future_seq_channels=None,
        future_seq_length=None,
        # norm=True,
    ) -> None:
        super().__init__()

        self.input_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size, latent_dim],
        )
        all_channels = seq_channels
        # self.seq_channels = seq_channels

        # include external features
        if future_seq_channels is not None and future_seq_length is not None:
            self.ex_enc = MLP(
                in_channels=future_seq_length,
                hidden_channels=[hidden_size, hidden_size, latent_dim],
            )
            all_channels += future_seq_channels
        if all_channels != target_seq_channels:
            self.ex_proj = nn.Linear(all_channels, target_seq_channels)
        else:
            self.ex_proj = nn.Identity()
            
        # self.rev = RevIN(seq_channels,affine=False)
            
        
    def forward(self, observed_data, future_features=None, **kwargs):
        # x = self.rev(observed_data, 'norm')
        x = self.input_enc(observed_data.permute(0, 2, 1))  # [bs, chn, hs]

        if future_features is not None:
            ff = self.ex_enc(future_features.permute(0, 2, 1))
            x = torch.concat([x, ff], dim=1)
        x = self.ex_proj(x.permute(0,2,1)) # [bs, hs, chn]
        # x = self.rev(x, 'denorm')
        return x

class IdentityConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        target_seq_length,
        target_seq_channels,
        future_seq_channels=None,
        future_seq_length=None,
        # norm=True,
    ) -> None:
        super().__init__()

    def forward(self, observed_data, future_features=None, **kwargs):
        # x = observed_data
        
        return observed_data

class FreqMLPConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        target_seq_length,
        target_seq_channels,
        future_seq_channels=None,
        future_seq_length=None,
        # norm=True,
    ) -> None:
        super().__init__()
        all_input_channel = seq_channels * seq_length

        # include external features
        if future_seq_channels is not None and future_seq_length is not None:
            all_input_channel += future_seq_channels * future_seq_length

        self.latent_dim = latent_dim
        # self.encode_tp = encode_tp
        self.input_enc = MLP(
            in_channels=all_input_channel,
            hidden_channels=[hidden_size, hidden_size, latent_dim],
        )

        # self.norm = norm
        self.seq_channels = seq_channels
        # if norm:
        #     self.mu_net = nn.Linear(latent_dim, 1 * seq_channels)
        #     self.std_net = nn.Linear(latent_dim, 1 * seq_channels)

    def forward(self, observed_data, future_features=None, **kwargs):
        x = observed_data
        x = dft(x)
        trajs_to_encode = x.flatten(1)  # (batch_size, input_ts, input_dim)

        # # TEST: frequency encoding
        # freq_component = torch.fft.rfft(observed_data, dim=1).flatten(1)
        # theta, phi = complex2sphere(freq_component.real, freq_component.imag)
        # trajs_to_encode = torch.concat([theta, phi, trajs_to_encode], dim=-1)

        if future_features is not None:
            ff = future_features
            trajs_to_encode = torch.concat([trajs_to_encode, ff.flatten(1)], axis=-1)
        out = self.input_enc(trajs_to_encode)

        # if self.norm:
        #     mu = self.mu_net(out).reshape((-1, 1, self.seq_channels))
        #     std = self.std_net(out).reshape((-1, 1, self.seq_channels))
        # else:
        #     mu, std = (
        #         torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device),
        #         torch.ones((x.shape[0], 1, x.shape[-1]), device=x.device),
        #     )
        return out


class DLinearConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        target_seq_length,
        target_seq_channels,
        future_seq_channels=None,
        future_seq_length=None,
        **kwargs,
        # norm=True,
    ) -> None:
        super().__init__()
        self.decompsition = series_decomp(kwargs.get("moving_avg", 25))
        self.Linear_Seasonal = nn.Linear(seq_length, target_seq_length)
        self.Linear_Trend = nn.Linear(seq_length, target_seq_length)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / seq_length) * torch.ones([target_seq_length, seq_length])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / seq_length) * torch.ones([target_seq_length, seq_length])
        )
        self.obs_enc = nn.Linear(seq_length * seq_channels, latent_dim)

    def forward(self, observed_data, future_features=None, **kwargs):
        seasonal_init, trend_init = self.decompsition(observed_data)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return dft(x.permute(0, 2, 1))


class RNNConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        future_seq_channels=None,
        future_seq_length=None,
        encode_tp=False,
        frequency=False,
    ) -> None:
        super().__init__()

    def forward(self):
        pass


class CNNConditioner(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        target_seq_length,
        target_seq_channels,
        future_seq_channels=None,
        future_seq_length=None,
        # norm=True,
    ) -> None:
        super().__init__()
        self.embedder = nn.Conv1d(
            in_channels=seq_channels,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        cond_seq_len = seq_length
        if future_seq_channels and future_seq_length:
            self.embedder_futurefeat = nn.Conv1d(
                in_channels=future_seq_channels,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            )
            cond_seq_len += future_seq_length

        self.timelinear = nn.Linear(cond_seq_len, target_seq_length)

    def forward(self, observed_data, future_features=None, **kwargs):
        x = self.embedder(observed_data.permute(0, 2, 1)).transpose(1, 2)
        if future_features is not None:
            ff = self.embedder_futurefeat(future_features.permute(0, 2, 1)).transpose(
                1, 2
            )
            x = torch.concat([x, ff], dim=1)
        x = self.timelinear(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x
