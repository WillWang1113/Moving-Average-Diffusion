import sys
import torch
from typing import List, Union, Tuple
from torch import is_complex, nn
from torchvision.ops import MLP
import numpy as np
import math

from src.layers.Autoformer_EncDec import series_decomp
from ..layers.MLP import Mlp
from ..layers.SelfAttention_Family import Attention
from ..layers.Embed import PatchEmbed


from ..layers.SelfAttention_Family import AttentionLayer, FullAttention
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from src.utils.filters import MovingAvgFreq
from src.utils.fourier import (
    real_imag_to_complex_freq,
    complex_freq_to_real_imag,
)
from .embedding import SinusoidalPosEmb
from .blocks import (
    UpBlock,
    DownBlock,
    MiddleBlock,
    Downsample,
    Upsample,
    ResidualBlock,
    CRELU,
    CSoftshrink,
    CMLP,
)

thismodule = sys.modules[__name__]


# TODO: build backbone function
def build_backbone(bb_config):
    bb_config_c = bb_config.copy()
    bb_net = getattr(thismodule, bb_config_c.pop("name"))
    return bb_net(**bb_config_c)


# class RMLPBackbone(nn.Module):
#     """
#     ## MLP backbone
#     """

#     def __init__(
#         self,
#         seq_channels: int,
#         seq_length: int,
#         d_model: int,
#         d_mlp: int,
#         # hidden_size: int,
#         latent_dim: int,
#         n_layers: int = 3,
#         # norm: bool = False,
#         dropout: float = 0.1,
#         **kwargs,
#     ) -> None:
#         """
#         * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
#         * `seq_length` is the number of timesteps in the time series.
#         * `hidden_size` is the hidden size
#         * `n_layers` is the number of MLP
#         """
#         super().__init__()

#         self.embedder = nn.Linear(seq_channels * (seq_length), d_model)
#         self.unembedder = nn.Linear(d_model, seq_channels * (seq_length))
#         self.pe = SinusoidalPosEmb(d_model)
#         # self.pe = GaussianFourierProjection(d_model)
#         self.net = nn.ModuleList(  # type: ignore
#             [
#                 MLP(
#                     in_channels=d_model,
#                     hidden_channels=[d_mlp, d_model],
#                     dropout=dropout,
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.seq_channels = seq_channels
#         self.seq_length = seq_length
#         if d_model != latent_dim:
#             self.con_linear = nn.Linear(latent_dim, d_model)
#         else:
#             self.con_linear = nn.Identity()

#     def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
#         fft_shape = x.shape[1]
#         x_real_imag = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
#         x = self.embedder(x_real_imag.flatten(1))
#         t = self.pe(t)
#         c = self.con_linear(condition)
#         x = x + t + c
#         # x = self.pe(x, t, use_time_axis=False) + c
#         for layer in self.net:
#             x = x + layer(x)

#             # x = x + t + c
#         # x = x + c
#         x = self.unembedder(x).reshape((-1, (self.seq_length), self.seq_channels))
#         # x = torch.nn.functional.softshrink(x)
#         x_re = x[:, :fft_shape, :]
#         x_im = torch.concat(
#             [
#                 torch.zeros_like(x[:, [0], :]),
#                 x[:, fft_shape:, :],
#                 torch.zeros_like(x[:, [0], :]),
#             ],
#             dim=1,
#         )
#         x = torch.stack([x_re, x_im], dim=-1)
#         return torch.view_as_complex(x)


class RMLPBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        d_model: int,
        d_mlp: int,
        # hidden_size: int,
        latent_dim: int,
        n_layers: int = 3,
        # norm: bool = False,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()

        self.embedder = nn.Linear(seq_length, d_model)
        self.unembedder = nn.Linear(d_model, seq_length)
        self.pe = SinusoidalPosEmb(d_model)
        # self.pe = GaussianFourierProjection(d_model)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=d_model,
                    hidden_channels=[d_mlp, d_model],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if d_model != latent_dim:
            self.con_linear = nn.Linear(latent_dim, d_model)
        else:
            self.con_linear = nn.Identity()
        self.con_pred = nn.Linear(d_model, seq_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        # c, _, _ = condition
        c = condition['latents']
        
        fft_shape = x.shape[1]
        x_real_imag = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        
        x = self.embedder(x_real_imag.permute(0,2,1))   # [bs, chn, d_model]
        t = self.pe(t).unsqueeze(1)     # [bs, 1 d_model]
        c = self.con_linear(c)  # [bs, chn, d_model]
        c_pred = self.con_pred(c).permute(0,2,1)
        
        
        x = x + t + c
        for layer in self.net:
            x = x + layer(x)    # [bs, chn, d_model]

        x = self.unembedder(x).permute(0,2,1)   # [bs, seq_len, chn]
        x = x + c_pred
        
        x_re = x[:, :fft_shape, :]
        x_im = torch.concat(
            [
                torch.zeros_like(x[:, [0], :]),
                x[:, fft_shape:, :],
                torch.zeros_like(x[:, [0], :]),
            ],
            dim=1,
        )
        x = torch.stack([x_re, x_im], dim=-1)
        return torch.view_as_complex(x)


class MLPBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        d_model: int,
        d_mlp: int,
        # hidden_size: int,
        latent_dim: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()

        self.embedder = nn.Linear(seq_length, d_model)
        self.unembedder = nn.Linear(d_model, seq_length)
        self.pe = SinusoidalPosEmb(d_model)
        # self.pe = GaussianFourierProjection(d_model)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=d_model,
                    hidden_channels=[d_mlp, d_model],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if d_model != latent_dim:
            self.con_linear = nn.Linear(latent_dim, d_model)
        else:
            self.con_linear = nn.Identity()
            
        self.con_pred = nn.Linear(d_model, seq_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        c = condition['latents']
        
        # # ! for test
        # x = torch.fft.rfft(x, dim=1, norm='ortho')
        # fft_len = x.shape[1]
        # x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        
        
        x = self.embedder(x.permute(0, 2, 1))
        t = self.pe(t).unsqueeze(1)
        c = self.con_linear(c)
        c_pred = self.con_pred(c).permute(0,2,1)
        
        
        x = (x + t) + c
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).permute(0, 2, 1)
        
        # # ! for test
        # x_re = x[:, :fft_len, :]
        # x_im = torch.concat(
        #     [
        #         torch.zeros_like(x[:, [0], :]),
        #         x[:, fft_len:, :],
        #         torch.zeros_like(x[:, [0], :]),
        #     ],
        #     dim=1,
        # )
        # x = torch.stack([x_re, x_im], dim=-1)
        # x = torch.view_as_complex(x)
        # x = torch.fft.irfft(x, dim=1, norm='ortho')
        
        x = x + c_pred
        return x

class FreqMLPBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        d_model: int,
        d_mlp: int,
        latent_dim: int,
        n_layers: int = 3,
        norm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()

        self.embedder = nn.Linear(
            seq_channels * (seq_length // 2 + 1), d_model, dtype=torch.cfloat
        )
        self.unembedder = nn.Linear(
            d_model, seq_channels * (seq_length // 2 + 1), dtype=torch.cfloat
        )
        self.pe1 = SinusoidalPosEmb(d_model)
        self.pe2 = SinusoidalPosEmb(d_model)
        self.net = nn.ModuleList(  # type: ignore
            [
                CMLP(
                    in_channels=d_model,
                    hidden_channels=[d_mlp, d_model],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if d_model != latent_dim:
            self.con_linear = nn.Linear(latent_dim, d_model, dtype=torch.cfloat)
        else:
            self.con_linear = nn.Identity()
        # self.sparse = CSoftshrink()
        # self.relu = CRELU()

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        x = self.embedder(x.flatten(1))
        t_re, t_im = self.pe1(t), self.pe2(t)
        t = torch.view_as_complex(torch.stack([t_re, t_im], dim=-1))
        c = self.con_linear(condition)
        c = c + t
        bias = x
        x = x * c
        # x = self.pe(x, t, use_time_axis=False) + c
        for layer in self.net:
            x = layer(x) * 0.02

        x = x + bias
        # x = self.sparse(x)
        x = self.unembedder(x).reshape(
            (-1, (self.seq_length // 2 + 1), self.seq_channels)
        )
        return x


# class SkipMLPBackbone(nn.Module):
#     """
#     ## MLP backbone
#     """

#     def __init__(
#         self,
#         input_seq_length: int,
#         seq_length: int,
#         hidden_size: int,
#         # latent_dim: int,
#         n_layers: int = 3,
#         **kwargs,
#     ) -> None:
#         """
#         * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
#         * `seq_length` is the number of timesteps in the time series.
#         * `hidden_size` is the hidden size
#         * `n_layers` is the number of MLP
#         """
#         super().__init__()

#         self.embedder = nn.Linear(input_seq_length, hidden_size)
#         self.unembedder = nn.Linear(hidden_size, seq_length)
#         # self.pe = SinusoidalPosEmb(hidden_size)
#         self.net = nn.ModuleList(  # type: ignore
#             [
#                 MLP(
#                     in_channels=hidden_size,
#                     hidden_channels=[hidden_size * 2, hidden_size],
#                     dropout=0.1,
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.seq_length = seq_length

#     def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
#         # x = self.embedder(x.flatten(1))
#         # t = self.pe(t)
#         # x = x + t
#         latents = self.embedder(condition.permute(0, 2, 1))
#         for layer in self.net:
#             latents = latents + layer(latents)

#         pred = self.unembedder(latents).permute(0, 2, 1)
#         return pred


class CIMLPBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        hidden_size: int,
        latent_dim: int,
        n_layers: int = 3,
        norm: bool = False,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()

        self.embedder = nn.Linear(seq_length, hidden_size)
        self.unembedder = nn.Linear(hidden_size, seq_length)
        self.pe = SinusoidalPosEmb(hidden_size)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if hidden_size != latent_dim:
            self.con_linear = nn.Linear(latent_dim, hidden_size)
        else:
            self.con_linear = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        x = self.embedder(x.permute(0, 2, 1))  # [bs, chn, hs]
        t = self.pe(t).unsqueeze(1)  # [bs, hs]
        x = x + t
        if condition is not None:
            c = self.con_linear(condition.permute(0, 2, 1))  # [bs, chn, hs]
            x = x + c
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).permute(0, 2, 1)
        return x


class DecompCIMLPBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        # seq_channels: int,
        seq_length: int,
        obs_seq_length,
        hidden_size: int,
        latent_dim: int,
        n_layers: int = 3,
        norm: bool = False,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()
        freqresp = [
            MovingAvgFreq(ks, seq_length=seq_length).Hw
            for ks in range(2, seq_length + 1)
        ]
        freqresp = torch.concat(freqresp)
        self.register_buffer("freqresp", freqresp)
        # self.rev = RevIN(seq_channels, affine=True)

        # self.low_freq_emb = nn.Linear(seq_length, seq_length)

        self.emb_f = nn.Linear(seq_length, hidden_size)
        self.emb_r = nn.Linear(seq_length, hidden_size)
        self.unembedder = nn.Linear(hidden_size, seq_length)
        self.pe = SinusoidalPosEmb(hidden_size)
        self.net_f = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.net_r = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        # self.seq_channels = seq_channels
        self.seq_length = seq_length
        self.con_lf_mlp = MLP(
            in_channels=obs_seq_length,
            hidden_channels=[hidden_size, hidden_size],
            dropout=dropout,
        )
        self.con_hf_mlp = MLP(
            in_channels=obs_seq_length,
            hidden_channels=[hidden_size, hidden_size],
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        # DECOMP
        x_filtered = self.freqresp[t] * real_imag_to_complex_freq(x)
        x_filtered = complex_freq_to_real_imag(x_filtered, self.seq_length)
        x_residual = x - x_filtered

        # OBS
        # x_obs = self.rev(condition)
        x_obs_freq = torch.fft.rfft(condition, dim=1, norm="ortho")
        x_obs_lf = x_obs_freq * self.freqresp[25]
        x_obs_hf = x_obs_freq - x_obs_lf

        x_obs_lf = torch.concat([x_obs_lf.real, x_obs_lf.imag[:, 1:-1, :]], dim=1)
        x_obs_hf = torch.concat([x_obs_hf.real, x_obs_hf.imag[:, 1:-1, :]], dim=1)

        x_obs_lf = self.con_lf_mlp(x_obs_lf.permute(0, 2, 1))
        x_obs_hf = self.con_hf_mlp(x_obs_hf.permute(0, 2, 1))  # [bs, chn, hs]

        t = self.pe(t).unsqueeze(1)
        # LEARN FILTERS
        # x_filtered = self.low_freq_emb(x_filtered.permute(0,2,1)).permute(0,2,1)
        x_residual = (
            self.emb_r(x_residual.permute(0, 2, 1)) + t + x_obs_hf
        )  # [bs, chn, hs]
        for layer in self.net_f:
            x_residual = x_residual + layer(x_residual)
        # x_residual = self.unembedder(x_residual).permute(0, 2, 1)

        x_filtered = self.emb_f(x_filtered.permute(0, 2, 1)) + t + x_obs_lf
        for layer in self.net_r:
            x_filtered = x_filtered + layer(x_filtered)

        x_cat = x_residual + x_filtered + x_obs_lf + x_obs_hf
        x = self.unembedder(x_cat).permute(0, 2, 1)
        # x = x_obs_pred
        return x


class CIAttentionBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        hidden_size: int,
        latent_dim: int,
        n_layers: int = 3,
        norm: bool = False,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()

        self.embedder = nn.Conv1d(
            in_channels=seq_channels,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.pe = SinusoidalPosEmb(hidden_size)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, attention_dropout=kwargs.get("dropout", 0.1)
                        ),
                        hidden_size,
                        kwargs.get("n_heads", 4),
                    ),
                    hidden_size,
                    kwargs.get("d_ff", None),
                    dropout=kwargs.get("dropout", 0.1),
                    # activation=kwargs['dropout']
                )
                for _ in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

        self.unembedder = nn.Linear(hidden_size, seq_channels)
        self.seq_channels = seq_channels
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        x = self.embedder(x.permute(0, 2, 1)).permute(0, 2, 1)
        t = self.pe(t).unsqueeze(1)
        x = x + t
        x, attns = self.encoder(x)
        x = x + condition
        x = self.unembedder(x)
        return x


# class SkipDLinearBackbone(nn.Module):
#     """
#     ## MLP backbone
#     """

#     def __init__(
#         self,
#         input_seq_length: int,
#         # seq_channels: int,
#         seq_length: int,
#         # d_model: int,
#         # latent_dim: int,
#         # n_layers: int = 3,
#         # norm: bool = False,
#         **kwargs,
#     ) -> None:
#         """
#         * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
#         * `seq_length` is the number of timesteps in the time series.
#         * `hidden_size` is the hidden size
#         * `n_layers` is the number of MLP
#         """
#         super().__init__()
#         self.seq_length = input_seq_length
#         self.pred_len = seq_length

#         self.decompsition = series_decomp(25)
#         self.Linear_Seasonal = nn.Linear(self.seq_length, self.pred_len)
#         self.Linear_Trend = nn.Linear(self.seq_length, self.pred_len)

#         self.Linear_Seasonal.weight = nn.Parameter(
#             (1 / self.seq_length) * torch.ones([self.pred_len, self.seq_length])
#         )
#         self.Linear_Trend.weight = nn.Parameter(
#             (1 / self.seq_length) * torch.ones([self.pred_len, self.seq_length])
#         )

#     def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
#         seasonal_init, trend_init = self.decompsition(condition)
#         seasonal_init, trend_init = (
#             seasonal_init.permute(0, 2, 1),
#             trend_init.permute(0, 2, 1),
#         )
#         seasonal_output = self.Linear_Seasonal(seasonal_init)
#         trend_output = self.Linear_Trend(trend_init)
#         out = seasonal_output + trend_output
#         return out.permute(0, 2, 1)


class ResNetBackbone(nn.Module):
    def __init__(
        self, seq_channels, seq_length, hidden_size, latent_dim, **kwargs
    ) -> None:
        super().__init__()
        self.block1 = ResidualBlock(seq_channels, hidden_size, hidden_size)
        self.block2 = torch.nn.AvgPool1d(2, 2)
        self.block3 = ResidualBlock(hidden_size, hidden_size, hidden_size)
        self.block4 = torch.nn.AvgPool1d(2, 2)
        self.fc_out = MLP(hidden_size * (seq_length // 4), [seq_channels * seq_length])
        self.time_emb = SinusoidalPosEmb(hidden_size)
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if hidden_size != latent_dim:
            self.con_linear = nn.Linear(latent_dim, hidden_size)
        else:
            self.con_linear = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition=None):
        t = self.time_emb(t)
        if condition is not None:
            c = self.con_linear(condition)
            t = t + c  # [bs, hiddensize]
        x = x.permute(0, 2, 1)
        x = self.block1(x, t)
        x = self.block2(x)
        x = self.block3(x, t)
        x = self.block4(x)
        x = self.fc_out(x.flatten(1)).reshape(-1, self.seq_length, self.seq_channels)
        return x


class UNetBackbone(nn.Module):
    """
    ## U-Net backbone
    """

    def __init__(
        self,
        seq_channels,
        seq_length,
        d_model: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False),
        n_blocks: int = 1,
        **kwargs,
    ):
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `n_channels` is number of channels in the initial feature map that we transform the multi-var time series into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * latent_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        self.seq_length = seq_length
        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project time series into feature map
        self.ts_proj = nn.Conv1d(seq_channels, d_model, kernel_size=3, padding=1)

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = SinusoidalPosEmb(d_model * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = d_model
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(in_channels, out_channels, d_model * 4, is_attn[i])
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            d_model * 4,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, d_model * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, d_model * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        # self.norm = nn.GroupNorm(8, latent_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv1d(in_channels, seq_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        """
        * `x` has shape `[batch_size, seq, in_channels]`
        * `t` has shape `[batch_size]`
        """

        # Transpose
        x = x.permute(0, 2, 1)

        # Get time-step embeddings
        t = self.time_emb(t)
        if condition is not None:
            t = t + condition

        # Get image projection
        x = self.ts_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]

        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        x = self.act(x)
        # x = self.act(self.norm(x))
        x = self.final(x)
        x = x.permute(0, 2, 1)

        return x


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        seq_length=96,
        patch_size=16,
        seq_channels=4,
        d_model=1152,
        n_layers=4,
        num_heads=16,
        mlp_ratio=4.0,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        learn_sigma=False,
        **kwargs,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = seq_channels
        self.out_channels = seq_channels * 2 if learn_sigma else seq_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        if kwargs.get("freq", False):
            self.freq = kwargs.get("freq", False)
            in_seq_length = seq_length // 2 + 1
            in_channels = seq_channels * 2
            self.out_channels = in_channels * 2 if learn_sigma else in_channels
        else:
            in_seq_length = seq_length
            in_channels = seq_channels

        self.x_embedder = PatchEmbed(
            in_seq_length, patch_size, in_channels, d_model, bias=True
        )
        self.t_embedder = TimestepEmbedder(d_model)
        self.y_embedder = nn.Identity()
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, d_model), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=kwargs.get("attn_drop", 0.0),
                    proj_drop=kwargs.get("proj_drop", 0.0),
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(d_model, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        seq: (N, seq_len, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        num_patch = x.shape[1]
        # h = w = int(x.shape[1] ** 0.5)
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], num_patch, p, c))
        x = torch.einsum("nhpc->nchp", x)
        x = x.reshape(shape=(x.shape[0], c, num_patch * p))
        return x.permute(0, 2, 1)

    def forward(self, x, t, condition):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if torch.is_complex(x):
            is_c = True
            x = torch.concat([x.real, x.imag], dim=-1)
        else:
            is_c = False
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = (seq_len / patch_size)
        t = self.t_embedder(t)  # (N, D)
        condition = self.y_embedder(condition)  # (N, D)
        c = t + condition  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if is_c:
            assert x.shape[-1] == 2
            x = torch.view_as_complex(x).unsqueeze(-1)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=np.float32).reshape(1, -1)
    # grid_w = np.arange(grid_size, dtype=np.float32)
    # grid = np.meshgrid(grid_h)  # here w goes first
    # grid = np.stack(grid, axis=0)

    # grid = grid.reshape([1, grid_size])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

# def DiT_XL_2(**kwargs):
#     return DiT(n_layers=28, d_model=1152, patch_size=2, num_heads=16, **kwargs)

# def DiT_XL_4(**kwargs):
#     return DiT(n_layers=28, d_model=1152, patch_size=4, num_heads=16, **kwargs)

# def DiT_XL_8(**kwargs):
#     return DiT(n_layers=28, d_model=1152, patch_size=8, num_heads=16, **kwargs)

# def DiT_L_2(**kwargs):
#     return DiT(n_layers=24, d_model=1024, patch_size=2, num_heads=16, **kwargs)

# def DiT_L_4(**kwargs):
#     return DiT(n_layers=24, d_model=1024, patch_size=4, num_heads=16, **kwargs)

# def DiT_L_8(**kwargs):
#     return DiT(n_layers=24, d_model=1024, patch_size=8, num_heads=16, **kwargs)

# def DiT_B_2(**kwargs):
#     return DiT(n_layers=12, d_model=768, patch_size=2, num_heads=12, **kwargs)

# def DiT_B_4(**kwargs):
#     return DiT(n_layers=12, d_model=768, patch_size=4, num_heads=12, **kwargs)

# def DiT_B_8(**kwargs):
#     return DiT(n_layers=12, d_model=768, patch_size=8, num_heads=12, **kwargs)

# def DiT_S_2(**kwargs):
#     return DiT(n_layers=12, d_model=384, patch_size=2, num_heads=6, **kwargs)

# def DiT_S_4(**kwargs):
#     return DiT(n_layers=12, d_model=384, patch_size=4, num_heads=6, **kwargs)

# def DiT_S_8(**kwargs):
#     return DiT(n_layers=12, d_model=384, patch_size=8, num_heads=6, **kwargs)


# DiT_models = {
#     'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
#     'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
#     'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
#     'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
# }
