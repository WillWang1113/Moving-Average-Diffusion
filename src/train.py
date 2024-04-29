import torch
import os
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# from src.models.diffusion import DDPM
from tqdm import tqdm

from src.models.diffusion import BaseDiffusion
from src.utils.fourier import idft


def loss_func(inputs, targets, loss_scale):
    err = (inputs - targets) ** 2
    err = torch.sum(err, dim=1)
    err = err * loss_scale
    return err.mean()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=5, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e5
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model, self.path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(
        self,
        diffusion: BaseDiffusion,
        epochs: int,
        lr: float = 1e-3,
        alpha: float = 0,
        early_stop: int = -1,
        device: str = "cpu",
        output_pth: str = "/home/user/data/FrequencyDiffusion/savings",
        **kwargs,
    ) -> None:
        self.diffusion = diffusion
        self.epochs = epochs
        self.alpha = alpha
        self.lr = lr
        self.optimizer = torch.optim.Adam(params=diffusion.get_params(), lr=lr)
        self.device = device
        self.writer = SummaryWriter()
        self.output_pth = output_pth
        self.early_stopper = (
            EarlyStopping(
                patience=early_stop, path=os.path.join(output_pth, "checkpoint.pt")
            )
            if early_stop > 0
            else None
        )
        self.train_counter = 0

    def train(self, train_dataloader, val_dataloader=None, epochs: int = None):
        if self.train_counter >= 1:
            print("Finetuning!")
            self.optimizer = torch.optim.Adam(
                params=self.diffusion.get_params(), lr=self.lr / 2
            )
        E = epochs if epochs is not None else self.epochs
        # torch.save(self.model, self.output_pth)
        for e in tqdm(range(E), ncols=50):
            # set train mode
            self._set_mode("train")

            train_loss = 0
            for batch in train_dataloader:
                for k in batch:
                    batch[k] = batch[k].to(self.device)

                # get target
                x = batch.pop("future_data")
                loss = self.diffusion.get_loss(x, condition=batch)

                train_loss += loss
                
                if self.alpha > 0:
                    for p in self.diffusion.backbone.parameters():
                        loss += 0.5 * self.alpha * (p * p).sum()

                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                self.optimizer.step()

            train_loss /= len(train_dataloader)

            if val_dataloader is not None:
                val_loss = self.eval(val_dataloader)
                self.writer.add_scalars(
                    "Loss", {"train": train_loss, "val": val_loss}, e
                )

                if self.early_stopper is not None:
                    self.early_stopper(val_loss, self.diffusion)
                    if self.early_stopper.early_stop:
                        print(f"Early Stop at Epoch {e+1}!\n")
                        break

            else:
                self.writer.add_scalar("Loss/train", train_loss, e)
                # self.writer.add_scalars('Loss', {"train": train_loss}, e)

        self.writer.close()
        if val_dataloader is not None:
            self.diffusion = torch.load(os.path.join(self.output_pth, "checkpoint.pt"))
        torch.save(self.diffusion, os.path.join(self.output_pth, "diffusion.pt"))
        self.train_counter += 1

    def eval(self, dataset):
        test_loss = 0
        self._set_mode("val")
        for batch in dataset:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            with torch.no_grad():
                # get target
                x = batch.pop("future_data")
                loss = self.diffusion.get_loss(x, condition=batch)
                test_loss += loss
        return test_loss / len(dataset)

    def test(self, test_dataloader, n_sample=50, scaler=None):
        self._set_mode("val")
        y_pred, y_real = [], []
        for batch in test_dataloader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            target = batch.pop("future_data")
            # target_shape = target.shape
            noise = self.diffusion.init_noise(target, batch, n_sample)
            s = self.diffusion.backward(noise, batch)
            samples = s.reshape((n_sample, target.shape[0], *s.shape[1:]))
            if self.diffusion.freq_kw["frequency"]:
                target = idft(target)
            y_pred.append(samples.detach().cpu())
            y_real.append(target.detach().cpu())
        y_pred = torch.concat(y_pred, dim=1)
        y_real = torch.concat(y_real)

        if scaler is not None:
            print("--" * 30, "inverse transform", "--" * 30)
            mean, std = scaler["data"]
            target = scaler["target"]
            y_pred = y_pred * std[target] + mean[target]
            y_real = y_real * std[target] + mean[target]

        return y_pred.numpy(), y_real.numpy()

    def _set_mode(self, mode):
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


# class Sampler:
#     def __init__(
#         self, diffusion, n_sample: int, scaler, device: str, freq_kw: dict
#     ) -> None:
#         self.diffusion = diffusion
#         self.n_sample = n_sample
#         self.device = device
#         self.scaler = scaler
#         self.freq_kw = freq_kw

#     def sample(self, test_dataloader):
#         self._set_mode("val")
#         y_pred, y_real = [], []
#         for batch in test_dataloader:
#             for k in batch:
#                 batch[k] = batch[k].to(self.device)
#             target = batch.pop("future_data")
#             # target_shape = target.shape
#             noise = self.diffusion.init_noise(target, batch, self.n_sample)
#             s = self.diffusion.backward(noise, batch)
#             samples = s.reshape((self.n_sample, *y_real.shape))
#             y_pred.append(samples.detach().cpu())
#             y_real.append(target.detach().cpu())
#         y_pred = torch.concat(y_pred, dim=1)
#         y_real = torch.concat(y_real)
#         if self.freq_kw["frequency"]:
#             y_real = idft(y_real, self.freq_kw["stereographic"])

#         if self.scaler is not None:
#             print("--" * 30, "inverse transform", "--" * 30)
#             mean, std = self.scaler["data"]
#             target = self.scaler["target"]
#             y_pred = y_pred * std[target] + mean[target]
#             y_real = y_real * std[target] + mean[target]
#         return y_pred, y_real

#     def _set_mode(self, mode):
#         if mode == "train":
#             self.diffusion.backbone.train()
#             if self.diffusion.conditioner is not None:
#                 self.diffusion.conditioner.train()
#         elif mode == "val":
#             self.diffusion.backbone.eval()
#             if self.diffusion.conditioner is not None:
#                 self.diffusion.conditioner.eval()
#         else:
#             raise ValueError("no such mode!")
