import torch
from src.models import conditioner, backbone, diffusion
from src.train import Trainer
from src.datamodule.dataset import syntheic_sine
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.fourier import idft

device = "cuda" if torch.cuda.is_available() else "cpu"

local_name = Path(__file__).name.split(".")[0]

T = 100
min_beta = 1e-4
max_beta = 2e-2
hidden_size = 512
latent_dim = 64
epochs = 50
frequency = True
train_dl, test_dl = syntheic_sine(frequency=frequency)
n_sample = 10

batch = next(iter(train_dl))
seq_length, seq_channels = (
    batch["observed_data"].shape[1],
    batch["observed_data"].shape[2],
)
target_seq_length, target_seq_channels = (
    batch["future_data"].shape[1],
    batch["future_data"].shape[2],
)



# bb_net = backbone.MLPBackbone(
#     seq_channels=target_seq_channels,
#     seq_length=target_seq_length,
#     hidden_size=hidden_size,
# )
# ! problem on unet on forecasting
bb_net = backbone.UNetBackbone(seq_channels, latent_channels=64, n_blocks=1)
# print(bb_net)
# bb_net = backbone.ResNetBackbone(seq_channels, seq_length=seq_length, latent_channels=128)
cond_net = None
# cond_net = conditioner.MLPConditioner(
#     seq_channels=seq_channels,
#     seq_length=seq_length,
#     hidden_size=hidden_size,
#     latent_dim=128 * 4,
#     # latent_dim=hidden_size,
# )
diff = diffusion.DDPM(
    backbone=bb_net,
    # conditioner=cond_net,
    T=T,
    min_beta=min_beta,
    max_beta=max_beta,
    device=device,
    frequency=frequency,
)

print("\n")
print("MODEL PARAM:")
print("Denoise Network:\t",
    sum([torch.numel(p) for p in bb_net.parameters()]))
# print('Condition Network:\t',
#     sum([torch.numel(p) for p in cond_net.parameters()])
# )
print("\n")
# params = list(bb_net.parameters()) + list(cond_net.parameters())
params = list(bb_net.parameters())
trainer = Trainer(
    diffusion=diff,
    optimizer=torch.optim.Adam(params, lr=1e-4),
    device=device,
    epochs=epochs,
    train_loss_fn=torch.nn.MSELoss(reduction="mean"),
)
trainer.train(train_dl)


if cond_net is None:
    # Generative task
    y_pred, y_real = trainer.test(train_dl, n_sample=n_sample)

    # # ! test !
    if frequency:
        y_real = idft(y_real)

    sample_pred = y_pred[0, :, 0, :].cpu().numpy()
    sample_real = y_real[0, :, 0].cpu().numpy()

    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    ax[0].plot(sample_real, label="real")
    ax[0].legend()
    for i in range(1, 4):
        ax[i].plot(sample_pred[..., i], c="black", alpha=0.75, label="generated")
        ax[i].legend()
    fig.suptitle(f"Backbone: {bb_net._get_name()}")
    fig.tight_layout()
    fig.savefig(f"assets/{local_name}.png")
else:
    # forecast task

    y_pred, y_real = trainer.test(test_dl, n_sample=n_sample)

    # # ! test !
    if frequency:
        y_real = idft(y_real)

    sample_pred = y_pred[0, :, 0, :].cpu().numpy()
    sample_real = y_real[0, :, 0].cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(sample_real, label="real")
    ax.legend()
    ax.plot(sample_pred, c="black", alpha=1 / n_sample)
    fig.suptitle(f"Backbone: {bb_net._get_name()}")
    fig.tight_layout()
    fig.savefig(f"assets/{local_name}.png")
