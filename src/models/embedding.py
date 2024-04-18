import torch
from torch import nn
import math


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.
    Courtesy of https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, latent_dim: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.d_model = latent_dim
        self.W = nn.Parameter(torch.randn((latent_dim + 1) // 2) * scale, requires_grad=False)

        self.dense = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        time_proj = timesteps[:, None] * self.W[None, :] * 2 * torch.pi
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # Slice to get exactly d_model
        t_emb = embeddings[:, : self.d_model]  # (batch_size, d_model)

        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)

        projected_emb: torch.Tensor = self.dense(t_emb)

        return x + projected_emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, latent_dim, is_random=False):
        super().__init__()
        assert (latent_dim % 2) == 0
        half_dim = latent_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, t):
        t = t.unsqueeze(1)
        freqs = t * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((t, fouriered), dim=-1)
        return fouriered



class SinusoidalPosEmb(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, latent_dim: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = latent_dim
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.SiLU()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb