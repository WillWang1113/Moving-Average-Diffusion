import torch
import torch.nn.functional as F
import lightning as L
from ..models.backbone import build_backbone
from ..models.conditioner import build_conditioner
import src.backbone


class DDPM(L.LightningModule):
    """MovingAvg Diffusion."""

    def __init__(
        self,
        backbone_config: dict,
        # conditioner_config: dict,
        noise_schedule: dict,
        norm=False,
        lr=2e-4,
        alpha=1e-5,
        T=1000,
        pred_x0=False,
    ) -> None:
        super().__init__()
        bb_class = getattr(src.backbone, backbone_config["name"])
        self.backbone = bb_class(**backbone_config)
        self.seq_length = self.backbone.seq_length
        self.norm = norm
        self.lr = lr
        self.alpha = alpha
        self.loss_fn = F.mse_loss
        self.T = T
        self.pred_x0 = pred_x0
        # ! Notice:
        # in the schedule.py, alphas for x_0 to x_t
        # in the schedule.py, alphas_bars for x_t to x_t+1

        # in this DDPM.py, attribute alpha_bars for x_0 to x_t
        # in this DDPM.py, attribute alphas for x_t to x_t+1

        self.register_buffer("alphas", noise_schedule["alphas"])
        self.register_buffer("betas", noise_schedule["betas"])
        self.register_buffer("alpha_bars", noise_schedule["alpha_bars"])
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.alpha)

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x)
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy, noise

    def training_step(self, batch, batch_idx):
        x = batch.pop("x")

        # if norm
        x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        loss = self._get_loss(x, batch, x_mean, x_std)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.pop("x")

        # if norm
        x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        loss = self._get_loss(x, batch, x_mean, x_std)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        assert self._sample_ready
        x_real = batch.pop("x")
        cond = batch.get("c", None)
        all_sample_x = []
        for _ in range(self.n_sample):
            
            x = self._init_noise(x_real)

            for t in range(self.T - 1, -1, -1):
                z = torch.randn_like(x)
                t_tensor = torch.tensor(t, device=x.device).repeat(x.shape[0])
                eps_theta, mu, std = self.backbone(x, t_tensor, cond)
                
                if self.pred_x0:
                    if t > 0:
                        mu_pred = (
                            torch.sqrt(self.alphas[t]) * (1 - self.alpha_bars[t - 1]) * x
                            + torch.sqrt(self.alpha_bars[t - 1]) * self.betas[t] * eps_theta
                        )
                        mu_pred = mu_pred / (1 - self.alpha_bars[t])
                    else:
                        mu_pred = eps_theta
                else:
                    mu_pred = (
                        x
                        - (1 - self.alphas[t])
                        / torch.sqrt(1 - self.alpha_bars[t])
                        * eps_theta
                    ) / torch.sqrt(self.alphas[t])
                if t == 0:
                    sigma = 0
                else:
                    sigma = torch.sqrt(
                        (1 - self.alpha_bars[t - 1])
                        / (1 - self.alpha_bars[t])
                        * self.betas[t]
                    )
                x = mu_pred + sigma * z

            # denorm
            if not self.norm:                    
                mu, std = 0.0, 1.0
                
            out_x = x * std + mu
            assert out_x.shape == x_real.shape
            all_sample_x.append(out_x)

        all_sample_x = torch.stack(all_sample_x)
            
        return all_sample_x

    def _get_loss(self, x, condition: dict = None, x_mean=None, x_std=None):
        batch_size = x.shape[0]
        cond = condition.get("c", None)
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy, noise = self.degrade(x, t)

        # eps_theta
        # c = self._encode_condition(condition)
        eps_theta, x_mean_hat, x_std_hat = self.backbone(x_noisy, t, cond)

        # compute loss
        if self.pred_x0:
            loss = F.mse_loss(eps_theta, x)
        else:
            loss = F.mse_loss(eps_theta, noise)

        # regularizatoin
        if x_mean is not None:
            loss += F.mse_loss(x_mean_hat, x_mean)
        if x_std is not None:
            loss += F.mse_loss(x_std_hat, x_std)

        return loss


    def _init_noise(
        self,
        x: torch.Tensor,
    ):
        return torch.randn_like(x)
    
        # shape = (self.n_sample, x.shape[0], x.shape[1], x.shape[2])
        # return torch.randn(shape, device=x.device)

    def config_sampling(self, n_sample, **kwargs):
        self.n_sample = n_sample
        self._sample_ready = True

    def _normalize(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-6)
        x_norm = (x - mean) / stdev
        return x_norm, (mean, stdev)
