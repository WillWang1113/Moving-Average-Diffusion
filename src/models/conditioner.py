import abc
import sys

import torch
from torch import nn
from torchvision.ops import MLP

thismodule = sys.modules[__name__]


# TODO: build backbone function
def build_conditioner(cn_config):
    cn_config_c = cn_config.copy()
    cn_net = getattr(thismodule, cn_config_c.pop("name"))
    return cn_net(**cn_config_c)


class BaseConditioner(nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        observed_data: torch.Tensor,
        observed_tp: torch.Tensor = None,
        future_features: torch.Tensor = None,
    ):
        raise NotImplementedError()


class MLPConditioner(BaseConditioner):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        future_seq_channels=None,
        future_seq_length=None,
        norm=True,
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

        self.norm = norm
        self.seq_channels = seq_channels

    def forward(self, observed_data, future_features=None, **kwargs):
        x = observed_data
        trajs_to_encode = x.flatten(1)  # (batch_size, input_ts, input_dim)

        if future_features is not None:
            ff = future_features
            trajs_to_encode = torch.concat([trajs_to_encode, ff.flatten(1)], axis=-1)
        out = self.input_enc(trajs_to_encode)

        return out


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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
