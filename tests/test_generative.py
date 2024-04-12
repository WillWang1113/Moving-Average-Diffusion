import torch
from src.models import backbone, diffusion
from src.train import Trainer
from src.datamodule.dataset import syntheic_sine
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

local_name = Path(__file__).name.split('.')[0]

T = 1000
min_beta = 1e-4
max_beta = 2e-2
hidden_size = 256
epochs = 1000
train_dl, test_dl = syntheic_sine()

batch = next(iter(train_dl))
input_ts, input_dim = batch["observed_data"].shape[1], batch["observed_data"].shape[2]
output_ts = batch["tp_to_predict"].shape[1]
tp_to_predict = batch["tp_to_predict"][0].squeeze()

diff = diffusion.DDPM(T=T, min_beta=min_beta, max_beta=max_beta, device=device)
bb = backbone.MLPBackbone(
    dim=input_dim, timesteps=input_ts, hidden_size=hidden_size, T=T
).to(device)
print('\n')
print("MODEL PARAM:")
print(sum([torch.numel(p) for p in bb.parameters()]))
print('\n')
trainer = Trainer(
    diffusion=diff,
    backbone=bb,
    optimizer=torch.optim.Adam(bb.parameters()),
    device=device,
    epochs=epochs,
    train_loss_fn=torch.nn.MSELoss(reduction='mean'),
)
trainer.train(train_dl)
y_pred, y_real = trainer.test(train_dl, n_sample=10)
print(y_pred.shape)
print(y_real.shape)

sample_pred = y_pred[0,:,0,:].cpu().numpy()
sample_real = y_real[0,:,0].cpu().numpy()

fig, ax = plt.subplots(2,2)
ax = ax.flatten()
ax[0].plot(sample_real, label='real')
ax[0].legend()
for i in range(1, 4):
    ax[i].plot(sample_pred[...,i], c='black', alpha = 0.75, label='generated')
    ax[i].legend()
fig.tight_layout()
fig.savefig(f'assets/{local_name}.png')