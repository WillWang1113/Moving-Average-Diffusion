import os
import random
from datetime import datetime
import time
import math
from gluonts.transform import (
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    MapTransformation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    ExpandDimArray,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.dataset.field_names import FieldName

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


class EarlyStopper:
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(
        self,
        # model: BaseModel,
        epochs: int = 10,
        lr: float = 1e-3,
        alpha: float = 0,
        early_stop: int = -1,
        val_step: int = 1,
        device: str = "cpu",
        output_pth: str = "/home/user/data/FrequencyDiffusion/savings",
        smoke_test: bool = False,
        exp_name="model",
    ) -> None:
        # model = model
        self.epochs = epochs
        self.alpha = alpha
        self.lr = lr

        self.device = device
        self.exp_name = exp_name
        self.exp_name += datetime.now().strftime("%Y%m%d%H%M%S")

        self.output_pth = output_pth
        self.early_stopper = (
            EarlyStopper(
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
        model = model.to(self.device)
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

                for n, p in model.named_parameters():
                    if n.__contains__('backbone') and n.__contains__('net'):
                        loss += self.alpha * torch.norm(p, p=1)
                # if self.alpha > 0:
                #     for p in model.get_params():
                #         loss += 0.5 * self.alpha * (p * p).sum()
                train_loss += loss

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                optimizer.step()
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
            if self.smoke_test:
                break
        writer.close()

        # save best model
        if val_dataloader is not None:
            print("load best model")
            best_model_path = os.path.join(self.output_pth, "checkpoint.pt")
            model.load_state_dict(torch.load(best_model_path))
        return model

    def validate(self, model: BaseModel, dataloader):
        test_loss = 0
        model.eval()
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            loss = model.validation_step(batch)
            test_loss += loss
        return test_loss / len(dataloader)

    def predict(self, model: BaseModel, dataloader):
        model.to(self.device)
        model.eval()
        all_pred, all_label = [], []
        for batch in tqdm(dataloader, ncols=100):
            for k in batch:
                batch[k] = batch[k].to(self.device)
            all_label.append(batch["future_data"].cpu())
            # with torch.autocast(device_type="cuda"):
            pred = model.predict_step(batch)
            all_pred.append(pred.cpu())
            if self.smoke_test:
                break
            # print(time.time()-start)
        return all_pred, all_label


def get_expname_(model_config, data_config):
    bb_name = model_config["bb_config"].get("name")
    df_name = model_config["diff_config"].get("name")
    n_out = data_config["pred_len"]
    name = bb_name + "_"
    diff_config = model_config["diff_config"]
    if df_name.__contains__("Freq") or df_name.__contains__("freq"):
        name += "freq_"
    else:
        name += "time_"

    name += f"norm_{diff_config['norm']}_diff_{diff_config['pred_diff']}_{diff_config['noise_schedule']}_fcsth_{n_out}"

    return name


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss"}):
    return {m: metrics[m].item() for m in select}


class ConcatDataset:
    def __init__(self, test_pairs, axis=-1) -> None:
        self.test_pairs = test_pairs
        self.axis = axis

    def _concat(self, test_pairs):
        for t1, t2 in test_pairs:
            yield {
                "target": np.concatenate([t1["target"], t2["target"]], axis=self.axis),
                "start": t1["start"],
            }
    
    def __len__(self):
        return len(self.test_pairs)

    def __iter__(self):
        yield from self._concat(self.test_pairs)


def create_splitter(past_length: int, future_length: int, mode: str = "train"):
    if mode == "train":
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1,
            min_past=past_length,
            min_future=future_length,
        )
    elif mode == "val":
        instance_sampler = ValidationSplitSampler(min_future=future_length)
    elif mode == "test":
        instance_sampler = TestSplitSampler()

    splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=past_length,
        future_length=future_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )
    return splitter


def create_transforms(
    num_feat_dynamic_real,
    num_feat_static_cat,
    num_feat_static_real,
):
    remove_field_names = []
    if num_feat_static_real == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if num_feat_dynamic_real == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not num_feat_static_cat > 0
            else []
        )
        + [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # ExpandDimArray(field=FieldName.TARGET, axis=0)  
        ]
    )
