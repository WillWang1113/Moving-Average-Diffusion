import numpy as np
import torch
from src.models.diffusion import BaseDiffusion
from torch.nn import AvgPool1d, functional

import matplotlib.pyplot as plt


class Sampler:
    def __init__(self, diffusion: BaseDiffusion, n_sample: int, scaler) -> None:
        self.diffusion = diffusion
        self.n_sample = n_sample
        self.device = "cuda"
        # self.device = diffusion.get_params()[0].device
        self.scaler = scaler
        if isinstance(diffusion, BaseDiffusion):
            self.freq_kw = diffusion.freq_kw

    def sample(self, test_dataloader, smoke_test=False, collect_all=False):
        # FOR PROBABILISITC MODELS
        self._set_mode("val")
        y_pred, y_real = [], []
        for batch in test_dataloader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            target = batch.pop("future_data")
            # target_shape = target.shape
            noise = self.diffusion.init_noise(target, batch, self.n_sample)
            samples = []
            for i in range(self.n_sample):
                s = self.diffusion.sample(noise[i], batch, collect_all)
                samples.append(s)
            samples = torch.stack(samples)
            print(samples.shape)
            # samples = s.reshape((self.n_sample, target.shape[0], *s.shape[1:]))

            y_pred.append(samples)
            y_real.append(target)
            if smoke_test:
                break
        y_pred = torch.concat(y_pred, dim=1).detach().cpu()
        y_real = torch.concat(y_real).detach().cpu()

        if self.scaler is not None:
            print("--" * 30, "inverse transform", "--" * 30)
            mean, std = self.scaler["data"]
            target = self.scaler["target"]
            y_pred = y_pred * std[target] + mean[target]
            y_real = y_real * std[target] + mean[target]
        return y_pred.numpy(), y_real.numpy()

    def predict(self, test_dataloader):
        # FOR DETERMINISTIC MODELS
        self._set_mode("val")
        y_pred, y_real = [], []
        for batch in test_dataloader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            target = batch.pop("future_data")
            # target_shape = target.shape
            s = self.diffusion(batch["observed_data"], batch["future_features"])
            y_pred.append(s)
            y_real.append(target)
        y_pred = torch.concat(y_pred).detach().cpu()
        y_real = torch.concat(y_real).detach().cpu()

        if self.scaler is not None:
            print("--" * 30, "inverse transform", "--" * 30)
            mean, std = self.scaler["data"]
            target = self.scaler["target"]
            y_pred = y_pred * std[target] + mean[target]
            y_real = y_real * std[target] + mean[target]
        return y_pred.numpy(), y_real.numpy()

    def _set_mode(self, mode):
        if isinstance(self.diffusion, BaseDiffusion):
            if mode == "train":
                self.diffusion.backbone.train()
                if self.diffusion.conditioner is not None:
                    self.diffusion.conditioner.train()
            elif mode == "val":
                self.diffusion.backbone.eval()
                if self.diffusion.conditioner is not None:
                    self.diffusion.conditioner.eval()
            else:
                raise ValueError("no such mode!")
        else:
            if mode == "train":
                self.diffusion.train()
            elif mode == "val":
                self.diffusion.eval()
            else:
                raise ValueError("no such mode!")


def plot_fcst(y_pred, y_real, save_name, kernel_size: list = None):

    fig, ax = plt.subplots(3, 3, figsize=[8,6])
    ax = ax.flatten()
    if isinstance(y_pred, np.ndarray):
        n_sample = y_pred.shape[0]
        bs = y_pred.shape[1]
        for k in range(len(ax)):
            choose = np.random.randint(0, bs)
            sample_real = y_real[choose, :, 0]
            sample_pred = y_pred[:, choose, :, 0].T
            ax[k].plot(sample_real, label="real")
            ax[k].plot(sample_pred, c="black", alpha=1 / n_sample)
            ax[k].legend()
            ax[k].set_title(f"sample no. {choose}")
    else:
        n_sample = y_pred[0].shape[0]
        choose = 99
        for k in range(len(ax)):
            sample_real = y_real[-k - 1][choose, :, 0]
            sample_pred = y_pred[-k - 1][:, choose, :, 0].T
            ax[k].plot(sample_real, label="real")
            ax[k].plot(sample_pred, c="black", alpha=1 / n_sample)
            ax[k].legend()
            ax[k].set_title(f"MA kernel size: {kernel_size[-k-1]}")
        fig.suptitle(f"sample no. {choose}")
    fig.tight_layout()
    fig.savefig(save_name)


def temporal_avg(y_pred: torch.Tensor, y_real: torch.Tensor, kernel_size, kind):
    all_res_pred, all_res_real = [], []
    for i in range(y_pred.shape[-1]):
        res_y_pred = y_pred[..., i]
        ks = kernel_size[i]
        avgpool = AvgPool1d(ks, ks)
        # res_y_real = torch.from_numpy(y_real)
        res_y_real = avgpool(y_real.permute(0, 2, 1)).permute(0, 2, 1)
        # res_y_real = res_y_real.numpy()
        assert kind in ["freq", "time"]
        if kind == "time":
            # res_y_pred: [n_sample, bs, ts, dim]
            res_y_pred = res_y_pred.flatten(end_dim=1)
            # res_y_pred: [n_sample * bs, ts, dim]
            res_y_pred = functional.interpolate(
                res_y_pred.permute(0, 2, 1),
                size=y_real.shape[1] - kernel_size[i] + 1,
            )
            res_y_pred = res_y_pred[..., :: kernel_size[i]].permute(0, 2, 1)
            assert res_y_pred.shape[1:] == res_y_real.shape[1:]
            res_y_pred = res_y_pred.reshape((-1, *res_y_real.shape))
        else:
            res_y_pred = res_y_pred[:, :, ks - 1 :: ks, :]
            assert res_y_pred.shape[2] == res_y_real.shape[1]
        all_res_pred.append(res_y_pred.numpy())
        all_res_real.append(res_y_real.numpy())
    return all_res_pred, all_res_real
