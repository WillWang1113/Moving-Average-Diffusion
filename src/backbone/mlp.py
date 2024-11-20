import sys

import torch
from torch import nn
from torchvision.ops import MLP

from src.models.blocks import RevIN

from ..models.embedding import SinusoidalPosEmb

thismodule = sys.modules[__name__]


# TODO: build backbone function
def build_backbone(bb_config):
    bb_config_c = bb_config.copy()
    bb_net = getattr(thismodule, bb_config_c.pop("name"))
    return bb_net(**bb_config_c)


class ConditionEmbed(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.norm = norm
        # self.embedder = nn.Linear(seq_length, hidden_size)
        if norm:
            self.rev = RevIN(seq_channels)
        self.input_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.pred_dec = torch.nn.Linear(hidden_size, latent_dim)
        self.dropout_prob = cond_dropout_prob

        # self.mean_dec = torch.nn.Linear(target_seq_length, 1)
        # self.std_dec = torch.nn.Linear(target_seq_length, 1)

        self.seq_channels = seq_channels

    # def forward(self, observed_data, **kwargs):

    #     if self.norm:
    #         x_norm = self.rev(observed_data, "norm")
    #     else:
    #         x_norm = observed_data

    #     latents = self.input_enc(x_norm.permute(0, 2, 1))
    #     latents = self.pred_dec(latents).permute(0, 2, 1)

    #     # if self.norm:
    #     # latents = self.rev(latents, "denorm")

    #     return latents
    #     # mean_pred = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
    #     # std_pred = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)

    #     # return {"latents": latents, "mean_pred": mean_pred, "std_pred": std_pred}

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
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, force_drop_ids=None, **kwargs):
        use_dropout = self.dropout_prob > 0

        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        latents = self.input_enc(x_norm.permute(0, 2, 1))
        latents = self.pred_dec(latents).permute(0, 2, 1)

        # if self.norm:
        # latents = self.rev(latents, "denorm")

        return latents
        # mean_pred = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        # std_pred = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)

        # return {"latents": latents, "mean_pred": mean_pred, "std_pred": std_pred}


class MLPBackbone(nn.Module):
    """
    ## MLP backbone for denoising a time-domian data
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        d_model: int,
        d_mlp: int,
        # hidden_size: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        cond_dropout_prob=0.5,
        cond_seq_len=None,
        cond_seq_chnl=None,
        norm=True,
        freq_denoise=False,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()
        self.dropout_prob = cond_dropout_prob
        self.freq_denoise = freq_denoise
        self.norm = norm

        # condition
        self.cond_embed = ConditionEmbed(
            seq_channels=cond_seq_chnl,
            seq_length=cond_seq_len,
            hidden_size=d_mlp // 2,
            latent_dim=d_model,
            norm=norm,
            cond_dropout_prob=cond_dropout_prob,
        )

        self.mean_dec = torch.nn.Linear(d_model, 1)
        self.std_dec = torch.nn.Linear(d_model, 1)

        # self.freq_denoise = freq_denoise
        self.embedder = nn.Linear(seq_length, d_model)
        self.unembedder = nn.Linear(d_model, seq_length)
        self.pe = SinusoidalPosEmb(d_model)
        self.net = nn.ModuleList(
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

    # def token_drop(self, labels, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = (
    #             torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
    #         )
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
    #     labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
    #     return labels

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor = None,
        train=True,
        force_drop_ids=None,
    ):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     condition = self.token_drop(condition, force_drop_ids)

        input_freq = False
        if torch.is_complex(x):
            input_freq = True
            fft_len = x.shape[1]
            x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        elif self.freq_denoise:
            x = torch.fft.rfft(x, dim=1, norm="ortho")
            fft_len = x.shape[1]
            x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        # print('input')
        # print(x.shape)
        x = self.embedder(x.permute(0, 2, 1))
        # print('latent')
        # print(x.shape)
        t = self.pe(t).unsqueeze(1)
        # print('t')
        # print(t.shape)
        # print('c')
        # print(c.shape)
        # c = self.con_linear(c)
        if condition is not None:
            c = self.cond_embed(condition, train, force_drop_ids).permute(0, 2, 1)
            x = x + (t + c)
        else:
            x = x + t
        # print('combine')
        # print(x.shape)·
        for layer in self.net:
            x = x + layer(x)

        # x_denorm = torch.where(self.cond_embed.drop_ids, x, self.cond_embed.rev(x, "denorm"))
        # print(self.cond_embed.rev.affine_bias)
        # print(self.cond_embed.rev.affine_weight)
        x_denorm = self.cond_embed.rev(x.permute(0, 2, 1), "denorm").permute(0, 2, 1)
        x_mean = self.mean_dec(x_denorm).permute(0, 2, 1)
        x_std = self.std_dec(x_denorm).permute(0, 2, 1)
        # print('x_mean')
        # print(x_mean.shape)
        # print(x_std.shape)
        # if self.norm:
        #     x_mean = self.cond_embed.rev(x_mean, 'denorm')
        #     x_std = self.cond_embed.rev(x_std, 'denorm')

        x = self.unembedder(x).permute(0, 2, 1)
        # print('out_x')
        # print(x.shape)

        if self.freq_denoise or input_freq:
            # ! for test
            x_re = x[:, :fft_len, :]
            x_im = torch.concat(
                [
                    torch.zeros_like(x[:, [0], :]),
                    x[:, fft_len:, :],
                    torch.zeros_like(x[:, [0], :]),
                ],
                dim=1,
            )
            x = torch.stack([x_re, x_im], dim=-1)
            x = torch.view_as_complex(x)
        if self.freq_denoise:
            x = torch.fft.irfft(x, dim=1, norm="ortho")
        # print(x.shape)

        # return 0

        return x, x_mean, x_std
