import torch
from src.models.diffusion import BaseDiffusion


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
            s = self.diffusion.backward(noise, batch, collect_all)
            samples = s.reshape((self.n_sample, target.shape[0], *s.shape[1:]))

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



