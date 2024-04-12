import torch
from torch import nn
from torchvision.ops import MLP
from .embedding import GaussianFourierProjection

class MLPBackbone(nn.Module):
    def __init__(
        self, dim: int, timesteps: int, hidden_size: int, T: int, num_layers: int = 10
    ) -> None:
        super(MLPBackbone, self).__init__()
        self.embedder = nn.Linear(dim * timesteps, hidden_size)
        self.unembedder = nn.Linear(hidden_size, dim * timesteps)
        self.pe = GaussianFourierProjection(hidden_size)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[1024, hidden_size],
                    # dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_noisy, t, conditions=None):
        
        x = self.embedder(x_noisy.flatten(1))
        x = self.pe(x, t, use_time_axis=False)
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).reshape(x_noisy.shape)
        return x
    
    
# class CNNBackbone(nn.Module):
    
