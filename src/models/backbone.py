import torch
from torch import nn
import math


class TimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_time: int, use_time_axis: bool = True):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_time, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_time, d_emb)
        self.use_time_axis = use_time_axis

    def forward(
        self, x: torch.Tensor, timesteps: torch.LongTensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        """Adds a time encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the time encoding should be added
            timesteps (torch.LongTensor): Tensor of shape (batch_size,) containing the current timestep for each sample in the batch

        Returns:
            torch.Tensor: Tensor with an additional time encoding
        """
        t_emb = self.embedding(timesteps)  # (batch_size, d_model)
        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        assert isinstance(t_emb, torch.Tensor)
        return x + t_emb


class MLPBackbone(nn.Module):
    def __init__(self, dim: int, timesteps: int, hidden_size: int, T: int) -> None:
        super(MLPBackbone, self).__init__()
        self.pe = TimeEncoding(dim, T)
        self.net = nn.Sequential(
            nn.Linear(dim * timesteps, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim * timesteps),
        )

    def forward(self, x_noisy, t):
        x = self.pe(x_noisy, t)
        x = self.net(x.flatten(1))
        return self.net(x).reshape(x_noisy.shape)
