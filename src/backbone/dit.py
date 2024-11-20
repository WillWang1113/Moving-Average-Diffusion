# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from ..layers.MLP import Mlp
from ..layers.SelfAttention_Family import Attention
from ..layers.Embed import PatchEmbed
from torchvision.ops import MLP

from src.models.blocks import RevIN


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
        self.pred_dec = torch.nn.Linear(hidden_size * seq_channels, latent_dim)
        self.dropout_prob = cond_dropout_prob
        self.seq_channels = seq_channels

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
        latents = latents.flatten(1)
        latents = self.pred_dec(latents)

        # if self.norm:
        # latents = self.rev(latents, "denorm")

        return latents
        # mean_pred = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        # std_pred = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)

        # return {"latents": latents, "mean_pred": mean_pred, "std_pred": std_pred}


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, cond_dim, hidden_size, dropout_prob):
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        self.embedding_table = Mlp(
            cond_dim, hidden_features=hidden_size, out_features=hidden_size
        )
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
        labels = torch.where(drop_ids.unsqueeze(1), torch.zeros_like(labels), labels)
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
        seq_channels=1,
        d_model=1152,
        n_layers=4,
        num_heads=16,
        mlp_ratio=4.0,
        cond_dropout_prob=0.5,
        cond_seq_len=None,
        cond_seq_chnl=None,
        norm=True,
        freq_denoise=False,
        learn_sigma=False,
        **kwargs,
    ):
        super().__init__()
        self.freq_denoise = freq_denoise
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
            in_seq_length, patch_size, seq_channels, d_model, bias=True
        )
        self.t_embedder = TimestepEmbedder(d_model)
        
        self.cond = False
        if (cond_seq_chnl is not None) and (cond_seq_len is not None):
            self.cond = True
            self.cond_embed = ConditionEmbed(
                seq_channels=cond_seq_chnl,
                seq_length=cond_seq_len,
                hidden_size=d_model * 2,
                latent_dim=d_model,
                norm=norm,
                cond_dropout_prob=cond_dropout_prob,
            )
        
            self.mean_dec = torch.nn.Linear(self.seq_length, 1)
            self.std_dec = torch.nn.Linear(self.seq_length, 1)
        
        # self.y_embedder = nn.Identity()
        # self.y_embedder = LabelEmbedder(cond_dim, d_model, cond_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, d_model), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, num_heads, mlp_ratio=mlp_ratio) for _ in range(n_layers)]
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
        if self.cond:
            nn.init.normal_(self.cond_embed.input_enc[0].weight, std=0.02)
            nn.init.normal_(self.cond_embed.input_enc[-2].weight, std=0.02)

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

    def forward(self, x, t, condition=None, train=True, force_drop_ids=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        input_freq = False
        if torch.is_complex(x):
            input_freq = True
            fft_len = x.shape[1]
            x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        elif self.freq_denoise:
            x = torch.fft.rfft(x, dim=1, norm="ortho")
            fft_len = x.shape[1]
            x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = (seq_len / patch_size)
        t = self.t_embedder(t)  # (N, D)

        if condition is not None:
            condition = self.cond_embed(condition, train, force_drop_ids)  # (N, D)
            c = t + condition  # (N, D)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size * out_channels)
        x = self.unpatchify(x)  # (N, seq_len, out_channels)
        
        if condition is not None:
            x_denorm = self.cond_embed.rev(x, "denorm") if self.cond_embed.norm else x
            x_mean = self.mean_dec(x_denorm.permute(0, 2, 1)).permute(0, 2, 1)
            x_std = self.std_dec(x_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x_mean, x_std = torch.zeros_like(x)[:,[0],:], torch.ones_like(x)[:,[0],:]

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
            
        # print('outshape')
        # print(x.shape)
        # print(x_mean.shape)
        # print(x_std.shape)

        return x, x_mean, x_std

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
    omega = np.arange(embed_dim // 2, dtype=np.float64)
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


def DiT_XL_2(**kwargs):
    return DiT(n_layers=28, d_model=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(n_layers=28, d_model=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(n_layers=28, d_model=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(n_layers=24, d_model=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(n_layers=24, d_model=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(n_layers=24, d_model=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(n_layers=12, d_model=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(n_layers=12, d_model=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(n_layers=12, d_model=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(n_layers=12, d_model=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(n_layers=12, d_model=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(n_layers=12, d_model=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
