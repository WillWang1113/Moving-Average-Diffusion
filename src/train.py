import torch
import os
import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
# from src.models.diffusion import DDPM

def loss_func(inputs, targets, loss_scale):
    err = (inputs - targets) **2
    err = torch.sum(err, dim=1)
    err = err * loss_scale
    return err.mean()
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
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
        diffusion,
        backbone: torch.nn.Module,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        train_loss_fn,
        alpha: float = 0,
        early_stop: int = -1,
        device: str = "cpu",
        inputs: list = [],
        target: list = [],
        output_pth: str = "./",
    ) -> None:
        self.diffusion = diffusion
        self.backbone = backbone
        self.epochs = epochs
        self.alpha = alpha
        self.train_loss_fn = train_loss_fn
        self.optimizer = optimizer
        self.device = device
        self.inputs = inputs
        self.target = target
        # self.writer = SummaryWriter(log_dir=f"runs/log")
        self.output_pth = os.path.join(output_pth, "checkpoint.pt")
        self.early_stopper = (
            EarlyStopping(patience=early_stop, path=self.output_pth)
            if early_stop > 0
            else None
        )

    def train(self, train_dataloader, val_dataloader=None, epochs: int = None):
        E = epochs if epochs is not None else self.epochs
        T = self.diffusion.T
        # torch.save(self.model, self.output_pth)
        for e in range(E):
            self.backbone.train()
            train_loss = 0

            for batch in train_dataloader:
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                    batch_size = batch[k].shape[0]
                t = torch.randint(0, T, (batch_size, )).to(self.device)
                noise = torch.randn_like(batch["future_data"]).to(self.device)
                x_noisy = self.diffusion.forward(batch["future_data"], t, noise)
                eps_theta = self.backbone(x_noisy, t)
                loss = self.train_loss_fn(eps_theta, noise)

                train_loss += loss
                for p in self.backbone.parameters():
                    loss += 0.5 * self.alpha * (p * p).sum()

                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                self.optimizer.step()

            train_loss /= len(train_dataloader)

            if val_dataloader is not None:
                val_loss = self.eval(val_dataloader)
                # self.writer.add_scalars('loss', {"train": train_loss}, e)
                # self.writer.add_scalars('loss', {"val": val_loss}, e)

                if e % 1 == 0:
                    print(
                        "[Epoch %d/%d] [Train loss %.3f] [Val loss %.3f]"
                        % (e, E, train_loss, val_loss)
                    )

                if self.early_stopper is not None:
                    self.early_stopper(val_loss, self.model)
                    if self.early_stopper.early_stop:
                        print(f"Early Stop at Epoch {e+1}!\n")
                        break

            else:
                if e % 1 == 0:
                    print("[Epoch %d/%d] [Train loss %.3f]" % (e, E, train_loss))
        if val_dataloader is not None:
            self.model.load_state_dict(torch.load(self.output_pth).state_dict())

    def eval(self, dataset):
        test_loss = 0
        self.backbone.eval()
        for batch in dataset:
            for k in batch:
                batch[k] = batch[k].to(self.device)

            with torch.no_grad():
                pred = self.model(
                    observed_data=batch["observed_data"],
                    tp_to_predict=batch["tp_to_predict"],
                    observed_tp=batch.get("observed_tp", None),
                    future_features=batch.get("future_features", None),
                )
                loss = self.train_loss_fn(pred, batch["future_data"])

                test_loss += loss
        return test_loss / len(dataset)

    def test(self, dataset, scaler=None, n_sample=1):
        self.backbone.eval()
        y_pred, y_real = [], []
        for batch in dataset:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            samples = []
            for _ in range(n_sample):
                noise = torch.randn_like(batch['future_data']).to(self.device)
                s = self.diffusion.backward(noise, self.backbone)
                samples.append(s)
            samples = torch.stack(samples, dim=-1)
            y_pred.append(samples)
            y_real.append(batch["future_data"])
        y_pred = torch.concat(y_pred)
        y_real = torch.concat(y_real)

        # inverse transform
        if scaler:
            print("--" * 30, "inverse transform", "--" * 30)
            mean, std = scaler["data"]
            target = scaler["target"]
            y_pred = y_pred * std[target] + mean[target]

        return y_pred, y_real
