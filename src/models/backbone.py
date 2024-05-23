import sys
from typing import List, Tuple, Union

import torch
from torch import nn
from torchvision.ops import MLP

from .blocks import (
    CMLP,
    ComplexRELU,
    DownBlock,
    Downsample,
    MiddleBlock,
    ResidualBlock,
    UpBlock,
    Upsample,
)
from .embedding import SinusoidalPosEmb

thismodule = sys.modules[__name__]


# TODO: build backbone function
def build_backbone(bb_config):
    bb_config_c = bb_config.copy()
    bb_net = getattr(thismodule, bb_config_c.pop("name"))
    return bb_net(**bb_config_c)


class MLPBackbone(nn.Module):
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
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super(MLPBackbone, self).__init__()

        self.embedder = nn.Linear(seq_channels * seq_length, hidden_size)
        self.unembedder = nn.Linear(hidden_size, seq_channels * seq_length)
        self.pe = SinusoidalPosEmb(hidden_size)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=0.1,
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
        x = self.embedder(x.flatten(1))
        t = self.pe(t)
        x = x * t
        if condition is not None:
            c = self.con_linear(condition)
            x = x + c
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).reshape((-1, self.seq_length, self.seq_channels))
        return x


class CMLPBackbone(torch.nn.Module):
    def __init__(
        self, seq_channels, seq_length, hidden_size, latent_dim, n_layers=1
    ) -> None:
        super().__init__()
        self.fft_len = seq_length
        self.embedder = nn.Linear(seq_channels * self.fft_len, hidden_size).to(
            torch.cfloat
        )
        self.unembedder = nn.Linear(hidden_size, seq_channels * self.fft_len).to(
            torch.cfloat
        )
        self.pe = SinusoidalPosEmb(hidden_size)
        self.net = nn.ModuleList(  # type: ignore
            [
                CMLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=0.1,
                )
                for _ in range(n_layers)
            ]
        )

        self.seq_channals = seq_channels
        self.seq_length = seq_length
        if hidden_size != latent_dim:
            self.con_linear = nn.Linear(latent_dim, hidden_size)
        else:
            self.con_linear = nn.Identity()

    def forward(self, x, t: torch.Tensor, condition=None):
        # Take in ffted data [bs, fft_len, channel]
        x = self.embedder(x.flatten(1))
        t = self.pe(t)
        x = x + t
        if condition is not None:
            c = self.con_linear(condition)
            x = x + c
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).reshape((-1, self.fft_len, self.seq_channals))
        # output ffted data [bs, fft_len, channel]
        return x


class FreqLinear(torch.nn.Module):
    def __init__(
        self, seq_channels, seq_length, hidden_size, latent_dim, n_layer=1
    ) -> None:
        super().__init__()
        rfft_len = seq_length // 2 + 1
        self.hidden_size = hidden_size
        self.embedder = nn.Linear(seq_channels, hidden_size)

        # freq compression
        self.freq_kernel = nn.Sequential()
        freq_in_channel = [rfft_len]
        for i in range(n_layer):
            freq_out_channel = freq_in_channel[i] // 2
            self.freq_kernel.append(
                nn.Linear(freq_in_channel[i], freq_out_channel).to(torch.cfloat)
            )
            self.freq_kernel.append(ComplexRELU())
            freq_in_channel.append(freq_out_channel)

        # channel compression
        self.channel_kernel = nn.Sequential()
        hidden_in_channel = [hidden_size]
        for i in range(n_layer):
            hidden_out_channel = hidden_in_channel[i] // 2
            self.channel_kernel.append(
                nn.Linear(hidden_in_channel[i], hidden_out_channel).to(torch.cfloat)
            )
            self.channel_kernel.append(ComplexRELU())
            hidden_in_channel.append(hidden_out_channel)

        # # channel extension
        # self.channel_kernel_up = nn.Sequential()
        # for i in range(n_layer):
        #     hidden_out_channel = hidden_in_channel[-i-2]
        #     self.channel_kernel_up.append(nn.Linear(hidden_in_channel[-i-1], hidden_out_channel).to(torch.cfloat))
        #     self.channel_kernel_up.append(ComplexRELU())

        # # freq extension
        # self.freq_kernel_up = nn.Sequential()
        # for i in range(n_layer):
        #     freq_out_channel = freq_in_channel[-i-2]
        #     self.freq_kernel_up.append(nn.Linear(freq_in_channel[-i-1], freq_out_channel).to(torch.cfloat))
        #     self.freq_kernel_up.append(ComplexRELU())

        self.linear = torch.nn.Linear(
            hidden_out_channel * freq_out_channel, hidden_size
        ).to(torch.cfloat)
        self.linear_out = torch.nn.Linear(hidden_size, rfft_len * seq_channels).to(
            torch.cfloat
        )
        self.pe = SinusoidalPosEmb(hidden_size)
        self.rfft_len = rfft_len
        self.seq_channels = seq_channels

        if hidden_size != latent_dim:
            self.con_linear = nn.Linear(latent_dim, hidden_size)
        else:
            self.con_linear = nn.Identity()

    def forward(self, x, t: torch.Tensor, condition=None):
        t = self.pe(t)
        if condition is not None:
            c = self.con_linear(condition)
            t = t + c  # [bs, hiddensize]

        x = self.embedder(x)  # [bs, seq_len, hidden_size]
        x = x + t.unsqueeze(1)

        x = torch.fft.rfft(x, norm="ortho", dim=1)  # [bs, rfft_len, hidden_size]
        x = self.freq_kernel(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # [bs, rfft_len, hidden_size]
        x = self.channel_kernel(x)  # [bs, rfft_len, kernel_size]

        # x = self.linear(x)  # [bs, rfft_len, dim]
        x = self.linear(x.flatten(1))
        # x = x + t
        x = self.linear_out(x).reshape(
            -1, self.rfft_len, self.seq_channels
        )  # [bs, rfft_len, dim]
        x = torch.fft.irfft(x, norm="ortho", dim=1)

        return x


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
        seq_channels: int,
        hidden_size: int = 64,
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

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project time series into feature map
        self.ts_proj = nn.Conv1d(seq_channels, hidden_size, kernel_size=3, padding=1)

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = SinusoidalPosEmb(hidden_size * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_size
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(in_channels, out_channels, hidden_size * 4, is_attn[i])
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
            hidden_size * 4,
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
                up.append(
                    UpBlock(in_channels, out_channels, hidden_size * 4, is_attn[i])
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, hidden_size * 4, is_attn[i]))
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
