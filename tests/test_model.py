import torch
from src.models import backbone, diffusion
from src.train import Trainer
from src.datamodule.dataset import syntheic_sine
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

local_name = Path(__file__).name.split('.')[0]

T = 200
min_beta = 1e-4
max_beta = 2e-2
hidden_size = 128
epochs = 500
train_dl, test_dl = syntheic_sine(2000)

batch = next(iter(train_dl))
input_ts, input_dim = batch["observed_data"].shape[1], batch["observed_data"].shape[2]
output_ts = batch["tp_to_predict"].shape[1]
tp_to_predict = batch["tp_to_predict"][0].squeeze()

diff = diffusion.DDPM(T=T, min_beta=min_beta, max_beta=max_beta, device=device)
bb = backbone.MLPBackbone(
    dim=input_dim, timesteps=input_ts, hidden_size=hidden_size, T=T
).to(device)
trainer = Trainer(
    diffusion=diff,
    backbone=bb,
    optimizer=torch.optim.Adam(bb.parameters()),
    device=device,
    epochs=epochs,
    train_loss_fn=torch.nn.L1Loss(reduction='sum'),
)
trainer.train(train_dl)
y_pred, y_real = trainer.test(test_dl, n_sample=20)

sample_pred = y_pred[0,:,0,:].cpu().numpy()
sample_real = y_real[0,:,0].cpu().numpy()

fig, ax = plt.subplots()
ax.plot(sample_real, label='real')
ax.plot(sample_pred, ls='--', c='black', alpha = 0.1)
ax.legend()
fig.savefig(f'assets/{local_name}.png')