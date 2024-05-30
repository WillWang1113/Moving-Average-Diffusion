import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

# from src.models.diffusion import DDPM
from tqdm import tqdm

from src.models.base import BaseModel
import numpy as np


def tensor_dict_to(batch_dict: dict, device: str):
    for k in batch_dict:
        if isinstance(batch_dict[k], dict):
            tensor_dict_to(batch_dict[k], device)
        else:
            batch_dict[k] = batch_dict[k].to(device)


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
        # model: BaseModel,
        epochs: int,
        lr: float = 1e-3,
        alpha: float = 0,
        early_stop: int = -1,
        val_step: int = 1,
        device: str = "cpu",
        output_pth: str = "/home/user/data/FrequencyDiffusion/savings",
        smoke_test: bool = False,
        **kwargs,
    ) -> None:
        # model = model
        self.epochs = epochs
        self.alpha = alpha
        self.lr = lr

        self.device = device
        self.exp_name = kwargs.get("exp_name")
        self.exp_name += datetime.now().strftime("%Y%m%d%H%M%S")

        self.output_pth = output_pth
        self.early_stopper = (
            EarlyStopping(
                patience=early_stop, path=os.path.join(output_pth, "checkpoint.pt")
            )
            if early_stop > 0
            else None
        )
        self.val_step = val_step
        self.smoke_test = smoke_test

    def fit(
        self,
        model: BaseModel,
        train_dataloader,
        val_dataloader=None,
    ):
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        writer = SummaryWriter(f"runs/{self.exp_name}")
        # torch.save(model, self.output_pth)
        for e in tqdm(range(self.epochs), ncols=100):
            # set train mode
            model.train()
            train_loss = 0
            for batch in train_dataloader:
                for k in batch:
                    batch[k] = batch[k].to(self.device)

                loss = model.train_step(batch)

                # if self.alpha > 0:
                #     for p in model.get_params():
                #         loss += 0.5 * self.alpha * (p * p).sum()
                train_loss += loss

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                optimizer.step()
                # for p in model.parameters():
                #     print(p[0][0])
                #     break
                if self.smoke_test:
                    break

            train_loss /= len(train_dataloader)
            if val_dataloader is not None:
                if e % self.val_step == 0:
                    val_loss = self.validate(model, val_dataloader)
                    writer.add_scalars(
                        "loss", {"train": train_loss, "val": val_loss}, e
                    )

                    if self.early_stopper is not None:
                        self.early_stopper(val_loss, model)
                        if self.early_stopper.early_stop:
                            print(f"Early Stop at Epoch {e+1}!\n")
                            break

            else:
                writer.add_scalar("train", train_loss, e)
                # self.writer.add_scalars('Loss', {"train": train_loss}, e)
        writer.close()

        # save best model
        if val_dataloader is not None:
            model = torch.load(os.path.join(self.output_pth, "checkpoint.pt"))
        torch.save(model, os.path.join(self.output_pth, "diffusion.pt"))

    def validate(self, model: BaseModel, dataloader):
        test_loss = 0
        model.eval()
        for batch in dataloader:
            tensor_dict_to(batch, self.device)
            loss = model.validation_step(batch)
            test_loss += loss
        return test_loss / len(dataloader)

    def predict(self, model: BaseModel, dataloader):
        model.eval()
        all_pred = []
        for batch in dataloader:
            tensor_dict_to(batch, self.device)
            pred = model.predict_step(batch)
            all_pred.append(pred)
            if self.smoke_test:
                break

        return all_pred


def get_expname_(config, bb_name, cn_name=None, df_name=None):
    name = bb_name + "_"
    diff_config = config["diff_config"]
    if df_name.__contains__('Freq') or df_name.__contains__('freq'): 
        name += "freq_"
    else:
        name += "time_"

    name += f"norm_{diff_config['norm']}_diff_{diff_config['pred_diff']}_"

    if (
        diff_config["noise_kw"]["min_beta"] == diff_config["noise_kw"]["max_beta"]
    ) and (diff_config["noise_kw"]["max_beta"] == 0):
        name += "cold_"
    else:
        name += "hot_"
    return name
