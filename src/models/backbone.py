import torch
from typing import List, Union, Tuple
from torch import nn
from torchvision.ops import MLP
from .embedding import GaussianFourierProjection, SinusoidalPosEmb
from .blocks import (
    UpBlock,
    DownBlock,
    MiddleBlock,
    Downsample,
    Upsample,
    ResidualBlock,
)


class MLPBackbone(nn.Module):
    """
    ## MLP backbone
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        hidden_size: int,
        n_layers: int = 3, **kwargs
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
        self.seq_channals = seq_channels
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        x = self.embedder(x.flatten(1))
        t = self.pe(t)
        x = x + t
        if condition is not None:
            assert x.shape == condition.shape
            x = x + condition
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).reshape((-1, self.seq_length, self.seq_channals))
        return x


# class FreqBackbone(nn.Module):
#     """
#     ## MLP backbone
#     """

#     def __init__(
#         self,
#         seq_channels: int,
#         seq_length: int,
#         hidden_size: int,
#         n_layers: int = 10,
#         stereographic=False,
#     ) -> None:
#         """
#         * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
#         * `seq_length` is the number of timesteps in the time series.
#         * `hidden_size` is the hidden size
#         * `n_layers` is the number of MLP
#         """
#         super().__init__()
#         self.embedder = nn.Linear(seq_channels * seq_length, hidden_size)
#         self.pe = GaussianFourierProjection(hidden_size)
#         self.net = nn.ModuleList(  # type: ignore
#             [
#                 MLP(
#                     in_channels=hidden_size,
#                     hidden_channels=[1024, hidden_size],
#                     dropout=0.1,
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.channals = seq_channels
#         self.length = seq_length
#         self.stereographic = stereographic
#         self.theta_out = nn.Linear(hidden_size, seq_channels * seq_length // 2)
#         self.phi_out = nn.Linear(
#             hidden_size, seq_channels * seq_length - seq_channels * seq_length // 2
#         )

#     def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
#         bs = x.shape[0]
#         x = self.embedder(x.flatten(1))
#         x = self.pe(x, t, use_time_axis=False)
#         if condition is not None:
#             assert x.shape == condition.shape
#             x = x + condition
#         for layer in self.net:
#             # x = layer(x)
#             x = x + layer(x)

#         theta = self.theta_out(x)
#         phi = self.phi_out(x)
#         theta = theta.reshape(bs, -1, self.channals)
#         phi = phi.reshape(bs, -1, self.channals)
#         return torch.concat((theta, phi), dim=1)


class ResNetBackbone(nn.Module):
    def __init__(
        self, seq_channels, seq_length, hidden_size, **kwargs
    ) -> None:
        super().__init__()
        self.block1 = ResidualBlock(seq_channels, hidden_size, hidden_size)
        # self.block2 = Downsample(hidden_size)
        self.block3 = ResidualBlock(hidden_size, hidden_size, hidden_size)
        self.fc_out = MLP(hidden_size * (seq_length // 2), [seq_channels * seq_length])
        self.time_emb = SinusoidalPosEmb(hidden_size)
        self.seq_channels = seq_channels
        self.seq_length = seq_length

    def forward(self, x, t, condition=None):
        t = self.time_emb(t)
        t = t + condition
        x = x.permute(0, 2, 1)
        x = self.block1(x, t)
        # x = self.block2(x, t)
        x = self.block3(x, t)
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
        n_blocks: int = 1, **kwargs
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
        self.ts_proj = nn.Conv1d(
            seq_channels, hidden_size, kernel_size=3, padding=1
        )

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
                    DownBlock(
                        in_channels, out_channels, hidden_size * 4, is_attn[i]
                    )
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
            up.append(
                UpBlock(in_channels, out_channels, hidden_size * 4, is_attn[i])
            )
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
