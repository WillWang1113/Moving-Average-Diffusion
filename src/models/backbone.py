import sys
import torch
from typing import List, Union, Tuple
from torch import nn
from torchvision.ops import MLP

from ..layers.SelfAttention_Family import AttentionLayer, FullAttention
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from src.utils.filters import MovingAvgFreq
from src.utils.fourier import (
    complex2sphere,
    dft,
    idft,
    real_imag_to_complex_freq,
    complex_freq_to_real_imag,
    sphere2complex,
)
from .embedding import SinusoidalPosEmb, GaussianFourierProjection
from .blocks import (
    CMLP,
    ComplexRELU,
    RevIN,
    UpBlock,
    DownBlock,
    MiddleBlock,
    Downsample,
    Upsample,
    ResidualBlock,
)

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
        d_model:int,
        d_mlp:int,
        # hidden_size: int,
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

        self.embedder = nn.Linear(seq_channels * seq_length, d_model)
        self.unembedder = nn.Linear(d_model, seq_channels * seq_length)
        self.pe = SinusoidalPosEmb(d_model)
        # self.pe = GaussianFourierProjection(d_model)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=d_model,
                    hidden_channels=[d_mlp, d_model],
                    dropout=dropout,
                ) for _ in range(n_layers)
            ])
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if d_model != latent_dim:
            self.con_linear = nn.Linear(latent_dim, d_model)
        else:
            self.con_linear = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
        x = self.embedder(x.flatten(1))
        t = self.pe(t)
        c = self.con_linear(condition)
        x = x+t + c
        # x = self.pe(x, t, use_time_axis=False) + c
        for layer in self.net:
            x = x + layer(x)
            
            # x = x + t + c
        # x = x + c
        x = self.unembedder(x).reshape(
            (-1, self.seq_length, self.seq_channels))
        return x


class FreqMLPBackbone(nn.Module):
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
        dropout: float = 0.,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()

        # self.embedder = nn.Linear(seq_channels * seq_length, hidden_size)
        # self.unembedder = nn.Linear(hidden_size, seq_channels * seq_length)
        self.pe = SinusoidalPosEmb(seq_length)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=seq_length,
                    hidden_channels=[seq_length, seq_length],
                    activation_layer=nn.SiLU
                    # dropout=dropout,
                ) for _ in range(n_layers)
            ])
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if seq_length != latent_dim:
            self.con_linear = nn.Linear(latent_dim, seq_length)
        else:
            self.con_linear = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
        # x = self.embedder(x.flatten(1))
        t = torch.sigmoid(self.pe(t))
        c = self.con_linear(condition).unsqueeze(-1)
        x = x * t.unsqueeze(-1)
        for layer in self.net:
            x = x.permute(0,2,1) + layer(x.permute(0,2,1))
            x = x.permute(0,2,1) + c
            # x = x + t + c
        # x = x + c
        # x = self.unembedder(x).reshape(
        #     (-1, self.seq_length, self.seq_channels))
        return x


class SkipMLPBackbone(nn.Module):
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

        self.embedder = nn.Linear(seq_channels * seq_length, hidden_size)
        self.unembedder = nn.Linear(hidden_size, seq_channels * seq_length)
        self.pe = SinusoidalPosEmb(hidden_size)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=0.1,
                ) for _ in range(n_layers)
            ])
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if hidden_size != latent_dim:
            self.con_linear = nn.Linear(latent_dim, hidden_size)
        else:
            self.con_linear = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
        x = self.embedder(x.flatten(1))
        t = self.pe(t)
        x = x + t
        for layer in self.net:
            x = x + layer(x)
        if condition is not None:
            c = self.con_linear(condition)
            x = x + c
        x = self.unembedder(x).reshape(
            (-1, self.seq_length, self.seq_channels))
        return x


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
                ) for _ in range(n_layers)
            ])
        self.seq_channels = seq_channels
        self.seq_length = seq_length
        if hidden_size != latent_dim:
            self.con_linear = nn.Linear(latent_dim, hidden_size)
        else:
            self.con_linear = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
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
        self.register_buffer('freqresp', freqresp)
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
                ) for _ in range(n_layers)
            ])
        
        self.net_r = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[hidden_size * 2, hidden_size],
                    dropout=dropout,
                ) for _ in range(n_layers)
            ])
        # self.seq_channels = seq_channels
        self.seq_length = seq_length
        self.con_lf_mlp = MLP(in_channels=obs_seq_length,
                              hidden_channels=[hidden_size, hidden_size],
                              dropout=dropout)
        self.con_hf_mlp = MLP(in_channels=obs_seq_length,
                              hidden_channels=[hidden_size, hidden_size],
                              dropout=dropout)

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
        
        # DECOMP
        x_filtered = self.freqresp[t] * real_imag_to_complex_freq(x)
        x_filtered = complex_freq_to_real_imag(x_filtered, self.seq_length)
        x_residual = x - x_filtered

        # OBS
        # x_obs = self.rev(condition)
        x_obs_freq = torch.fft.rfft(condition, dim=1, norm='ortho')
        x_obs_lf = x_obs_freq * self.freqresp[25]
        x_obs_hf = x_obs_freq - x_obs_lf

        x_obs_lf = torch.concat([x_obs_lf.real, x_obs_lf.imag[:, 1:-1, :]],
                                dim=1)
        x_obs_hf = torch.concat([x_obs_hf.real, x_obs_hf.imag[:, 1:-1, :]],
                                dim=1)

        x_obs_lf = self.con_lf_mlp(x_obs_lf.permute(0, 2, 1))
        x_obs_hf = self.con_hf_mlp(x_obs_hf.permute(0, 2,
                                                    1))  # [bs, chn, hs]

        t = self.pe(t).unsqueeze(1)
        # LEARN FILTERS
        # x_filtered = self.low_freq_emb(x_filtered.permute(0,2,1)).permute(0,2,1)
        x_residual = self.emb_r(x_residual.permute(0, 2, 1)) + t + x_obs_hf  # [bs, chn, hs]
        for layer in self.net_f:
            x_residual = x_residual + layer(x_residual)
        # x_residual = self.unembedder(x_residual).permute(0, 2, 1)

        x_filtered = self.emb_f(x_filtered.permute(0, 2, 1)) + t + x_obs_lf 
        for layer in self.net_r:
            x_filtered = x_filtered + layer(x_filtered)
        
        x_cat = x_residual + x_filtered + x_obs_lf + x_obs_hf
        x = self.unembedder(x_cat).permute(0,2,1)
        # x = x_obs_pred
        return x


# class RevCIMLPBackbone(nn.Module):
#     """
#     ## MLP backbone
#     """

#     def __init__(
#         self,
#         seq_channels: int,
#         seq_length: int,
#         obs_seq_len: int,
#         # obs_seq_chn: int,
#         hidden_size: int,
#         latent_dim: int,
#         n_layers: int = 3,
#         norm: bool = False,
#         **kwargs,
#     ) -> None:
#         """
#         * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
#         * `seq_length` is the number of timesteps in the time series.
#         * `hidden_size` is the hidden size
#         * `n_layers` is the number of MLP
#         """
#         super().__init__()
#         self.seq_length = seq_length
#         # self.rev = RevIN(seq_channels, affine=False)
#         self.obs_seq_len = obs_seq_len
#         # self.obs_seq_chn = obs_seq_chn
#         self.length_ratio = (seq_length + obs_seq_len)/obs_seq_len
#         self.obs_enc = nn.Linear((obs_seq_len//2 + 1), seq_length//2 + 1).to(torch.cfloat) # complex layer for frequency upcampling]

#         self.pe = SinusoidalPosEmb(hidden_size)
#         self.emb = nn.Linear(seq_channels, hidden_size).to(torch.cfloat)
#         self.unemb = nn.Linear(hidden_size, seq_channels).to(torch.cfloat)
#         maf = [MovingAvgFreq(ks, seq_length).Hw for ks in range(2, seq_length + 1)]
#         freqresp = torch.concat(maf)
#         self.register_buffer('freqresp', freqresp)
#         self.denoise_clinear = nn.ModuleList()
#         for l in range(n_layers):
#             self.denoise_clinear.append(nn.Linear(seq_length//2 + 1, seq_length//2 + 1, dtype=torch.cfloat))

#     def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):

#         # FOR LOOKBACK WINDOW
#         x_mean = torch.mean(condition, dim=1, keepdim=True)
#         x_var=torch.var(condition, dim=1, keepdim=True)+ 1e-5
#         x_obs = condition - x_mean
#         # print(x_var)
#         x_obs = x_obs / torch.sqrt(x_var)

#         # x_obs = self.rev(condition, 'norm')
#         x_obs_freq = torch.fft.rfft(x_obs, dim=1, norm='ortho')
#         # x_obs_freq = self.freqresp[t] * x_obs_freq
#         x_obs_freq = self.obs_enc(x_obs_freq.permute(0,2,1)).permute(0,2,1)
#         # x_obs_freq = torch.relu(x_obs_freq.real) + 1.0j * torch.relu(x_obs_freq.imag)

#         t = self.pe(t) # [bs, hs]
#         x = real_imag_to_complex_freq(x) # [bs, seq_len // 2 + 1, ch]
#         x = self.emb(x) + t.unsqueeze(1) # [bs, seq_len // 2 + 1, hs]
#         for layer in self.denoise_clinear:
#             x = layer(x.permute(0,2,1)).permute(0,2,1)
#             x = x*torch.sigmoid(x)
#         x = self.unemb(x)
#         x = x + x_obs_freq
#         x = x * x_var
#         x[:, [0],:] += x_mean
#         x[:, [0],:].imag *= 0
#         x = torch.concat([x.real, x.imag[:,1:-1,:]], dim=1)

#         return x


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
                        FullAttention(False,
                                      attention_dropout=kwargs.get(
                                          "dropout", 0.1)),
                        hidden_size,
                        kwargs.get("n_heads", 4),
                    ),
                    hidden_size,
                    kwargs.get("d_ff", None),
                    dropout=kwargs.get("dropout", 0.1),
                    # activation=kwargs['dropout']
                ) for _ in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

        self.unembedder = nn.Linear(hidden_size, seq_channels)
        self.seq_channels = seq_channels
        self.seq_length = seq_length

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
        x = self.embedder(x.permute(0, 2, 1)).permute(0, 2, 1)
        t = self.pe(t).unsqueeze(1)
        x = x + t
        x, attns = self.encoder(x)
        x = x + condition
        x = self.unembedder(x)
        return x


class ResNetBackbone(nn.Module):

    def __init__(self, seq_channels, seq_length, hidden_size, latent_dim,
                 **kwargs) -> None:
        super().__init__()
        self.block1 = ResidualBlock(seq_channels, hidden_size, hidden_size)
        self.block2 = torch.nn.AvgPool1d(2, 2)
        self.block3 = ResidualBlock(hidden_size, hidden_size, hidden_size)
        self.block4 = torch.nn.AvgPool1d(2, 2)
        self.fc_out = MLP(hidden_size * (seq_length // 4),
                          [seq_channels * seq_length])
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
        x = self.fc_out(x.flatten(1)).reshape(-1, self.seq_length,
                                              self.seq_channels)
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
        self.ts_proj = nn.Conv1d(seq_channels,
                                 hidden_size,
                                 kernel_size=3,
                                 padding=1)

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
                    DownBlock(in_channels, out_channels, hidden_size * 4,
                              is_attn[i]))
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
                    UpBlock(in_channels, out_channels, hidden_size * 4,
                            is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(in_channels, out_channels, hidden_size * 4,
                        is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        # self.norm = nn.GroupNorm(8, latent_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv1d(in_channels,
                               seq_channels,
                               kernel_size=3,
                               padding=1)

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor = None):
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
