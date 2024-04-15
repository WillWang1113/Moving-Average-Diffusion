import torch
from torch import nn
from torchvision.ops import MLP


class MLPConditioner(nn.Module):
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
        all_input_channel = seq_channels * seq_length
        # include observed time
        if encode_tp:
            all_input_channel += seq_length

        # include external features
        if future_seq_channels is not None and future_seq_length is not None:
            all_input_channel += future_seq_channels * future_seq_length

        # # TEST: frequency encoding
        # all_input_dim += (seq_length // 2 + 1) * seq_channels * 2

        self.latent_dim = latent_dim
        self.encode_tp = encode_tp
        self.input_enc = MLP(
            in_channels=all_input_channel, hidden_channels=[hidden_size, latent_dim]
        )

    # def forward(
    #     self, observed_data, future_data=None, observed_tp=None, future_features=None
    # ):
    #     return self.encode_input(
    #         observed_data=observed_data,
    #         observed_tp=observed_tp,
    #         future_features=future_features,
    #     )

    def forward(self, observed_data, observed_tp=None, future_features=None, **kwargs):
        trajs_to_encode = observed_data.flatten(1)  # (batch_size, input_ts, input_dim)

        # # TEST: frequency encoding
        # freq_component = torch.fft.rfft(observed_data, dim=1).flatten(1)
        # theta, phi = complex2sphere(freq_component.real, freq_component.imag)
        # trajs_to_encode = torch.concat([theta, phi, trajs_to_encode], dim=-1)

        if self.encode_tp:
            trajs_to_encode = torch.cat(
                (trajs_to_encode, observed_tp.flatten(1)), dim=-1
            )
        if future_features is not None:
            trajs_to_encode = torch.concat(
                [trajs_to_encode, future_features.flatten(1)], axis=-1
            )
        return self.input_enc(trajs_to_encode)


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
