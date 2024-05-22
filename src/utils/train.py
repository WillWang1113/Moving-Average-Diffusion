import os
import random
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

# from src.models.diffusion import DDPM
from tqdm import tqdm

from src.models.diffusion import BaseDiffusion
import numpy as np


def setup_seed(fix_seed=9):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        val_step:int=1,
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
        self.exp_name = kwargs.get("exp_name")
        self.exp_name +=  datetime.now().strftime("%Y%m%d%H%M%S")

        self.output_pth = output_pth
        self.early_stopper = (
            EarlyStopping(
                patience=early_stop, path=os.path.join(output_pth, "checkpoint.pt")
            )
            if early_stop > 0
            else None
        )
        self.val_step = val_step

    def train(self, train_dataloader, val_dataloader=None, epochs: int = None):
        writer = SummaryWriter(f"runs/{self.exp_name}")
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
                    for p in self.diffusion.get_params():
                        loss += 0.5 * self.alpha * (p * p).sum()

                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                self.optimizer.step()

            train_loss /= len(train_dataloader)

            if val_dataloader is not None:
                if e % self.val_step == 0:
                    val_loss = self.eval(val_dataloader)
                    writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, e)

                    if self.early_stopper is not None:
                        self.early_stopper(val_loss, self.diffusion)
                        if self.early_stopper.early_stop:
                            print(f"Early Stop at Epoch {e+1}!\n")
                            break

            else:
                writer.add_scalar("train", train_loss, e)
                # self.writer.add_scalars('Loss', {"train": train_loss}, e)

        writer.close()
        if val_dataloader is not None:
            self.diffusion = torch.load(os.path.join(self.output_pth, "checkpoint.pt"))
        torch.save(self.diffusion, os.path.join(self.output_pth, "diffusion.pt"))

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


def get_expname_(config, bb_name, cn_name=None, df_name=None):
    name = bb_name + "_"
    diff_config = config["diff_config"]
    if diff_config["freq_kw"]["frequency"]:
        name += "freq_"
    else:
        name += "time_"

    name += f"norm_{diff_config['norm']}_diff_{diff_config['fit_on_diff']}_"

    if (
        diff_config["noise_kw"]["min_beta"] == diff_config["noise_kw"]["max_beta"]
    ) and (diff_config["noise_kw"]["max_beta"] == 0):
        name += "cold_"
    else:
        name += "hot_"
    return name
