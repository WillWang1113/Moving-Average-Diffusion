from typing import Optional, Union
import warnings
import numpy as np
import torch
import tqdm

from torch import nn
import lightning as L

import copy
from scipy import linalg

# mqloss, mae, mse are from neuralforecast.losses.numpy
def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Multi-Quantile loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    $$ \mathrm{MQL}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau}) $$

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution $\mathbf{\hat{F}}_{\\tau}$ with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    $$ \mathrm{CRPS}(y_{\\tau}, \mathbf{\hat{F}}_{\\tau}) = \int^{1}_{0} \mathrm{QL}(y_{\\tau}, \hat{y}^{(q)}_{\\tau}) dq $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `quantiles`: numpy array,(n_quantiles). Quantiles to estimate from the distribution of y.<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `mqloss`: numpy array, (single value).

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """
    if weights is None:
        weights = np.ones(y.shape)

    # _metric_protections(y, y_hat, weights)
    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    mqloss = quantiles * sq + (1 - quantiles) * s1_q

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss


def mae(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Mean Absolute Error

    Calculates Mean Absolute Error between
    `y` and `y_hat`. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    $$ \mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} |y_{\\tau} - \hat{y}_{\\tau}| $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `mae`: numpy array, (single value).
    """
    # _metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    if weights is not None:
        mae = np.average(
            delta_y[~np.isnan(delta_y)], weights=weights[~np.isnan(delta_y)], axis=axis
        )
    else:
        mae = np.nanmean(delta_y, axis=axis)

    return mae

def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Mean Squared Error

    Calculates Mean Squared Error between
    `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series.

    $$ \mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2} $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `mse`: numpy array, (single value).
    """
    # _metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(
            delta_y[~np.isnan(delta_y)], weights=weights[~np.isnan(delta_y)], axis=axis
        )
    else:
        mse = np.nanmean(delta_y, axis=axis)

    return mse



def calculate_metrics(
    y_pred,
    y_real,
    quantiles=(np.arange(9) + 1) / 10,
    # normalize=False
):
    y_pred = y_pred.cpu().numpy()
    y_real = y_real.cpu().numpy()

    assert y_pred.ndim == 4
    assert y_pred[0].shape == y_real.shape
    print(y_pred.shape)
    y_pred_point = np.mean(y_pred, axis=0)
    y_pred_q = np.quantile(y_pred, quantiles, axis=0)
    print(y_pred_q.shape)
    y_pred_q = np.transpose(y_pred_q, (1, 2, 3, 0))
    print(y_pred_q.shape)

    MAE = mae(y_real, y_pred_point)
    MSE = mse(y_real, y_pred_point)
    MQL = mqloss(y_real, y_pred_q, quantiles=np.array(quantiles))
    return (MAE, MSE, MQL), y_pred_q, y_pred_point


def consistency_error(y_pred, y_real, kernel_size):
    # input: low-res data
    lr_y_real = torch.nn.functional.avg_pool1d(
        y_real.permute(0, 2, 1), kernel_size, kernel_size
    ).permute(0, 2, 1)

    # downsampled high-res output
    lr_y_pred = torch.nn.functional.avg_pool1d(
        y_pred.permute(0, 2, 1), kernel_size, kernel_size
    ).permute(0, 2, 1)

    return torch.nn.functional.mse_loss(lr_y_pred, lr_y_real).cpu().numpy()


def log_spectral_distance(y_pred, y_real):
    fft_sig1 = torch.fft.rfft(y_pred, dim=1)
    fft_sig2 = torch.fft.rfft(y_real, dim=1)

    # Convert to log-magnitude spectrum
    log_fft_sig1 = torch.log(torch.abs(fft_sig1) + 1e-5)
    log_fft_sig2 = torch.log(torch.abs(fft_sig2) + 1e-5)
    SE = (log_fft_sig1 - log_fft_sig2) ** 2
    # print(log_fft_sig2)
    # Compute the log-spectral distance
    lsd = torch.sqrt(torch.sum(SE, dim=1))
    return lsd.mean().cpu().numpy()


def sr_metrics(y_pred, y_real, kernel_size, autoencoder):
    y_pred = y_pred.squeeze(0)
    con_err = consistency_error(y_pred, y_real, kernel_size)
    lsd = log_spectral_distance(y_pred, y_real)
    mse_cal = mse(y_pred.cpu().numpy(), y_real.cpu().numpy())
    cfid = context_fid(y_pred, y_real, autoencoder)
    return mse_cal, lsd, con_err, cfid


class ConvEncoder(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[64, 128, 256], **kwargs
    ):
        super().__init__()
        current_seq_len = seq_len
        in_dim = seq_dim

        # Build Encoder
        modules = []
        for h_dim in hidden_size_list:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_dim = h_dim
            current_seq_len = current_seq_len // 2
        self.latent_seq_len = current_seq_len
        self.encoder = nn.Sequential(*modules)
        self.linear = nn.Linear(hidden_size_list[-1] * current_seq_len, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x.flatten(start_dim=1))
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[256, 128, 64], **kwargs
    ):
        super().__init__()
        latent_seq_len = copy.deepcopy(seq_len)
        for i in range(len(hidden_size_list)):
            latent_seq_len = latent_seq_len // 2
        self.latent_seq_len = latent_seq_len

        self.decoder_input = nn.Linear(latent_dim, hidden_size_list[0] * latent_seq_len)
        self.hidden_size_list = hidden_size_list

        modules = []
        for i in range(len(hidden_size_list) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_size_list[i],
                        hidden_size_list[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_size_list[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_size_list[-1],
                hidden_size_list[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(hidden_size_list[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size_list[-1],
                out_channels=seq_dim,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.hidden_size_list[0], self.latent_seq_len)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


class VanillaAE(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list: list = [64, 128, 256],
        # beta: float = 1e-3,
        # lr: float = 1e-3,
        # weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.encoder = ConvEncoder(**self.hparams)
        self.hparams.hidden_size_list.reverse()
        self.decoder = ConvDecoder(**self.hparams)

        # self.fc_mu = MLP(latent_dim, [latent_dim])
        # self.fc_logvar = MLP(latent_dim, [latent_dim])

    def encode(self, x):
        x = x.permute(0, 2, 1)
        latents = self.encoder(x)
        # la = self.fc_mu(latents)
        # logvar = self.fc_logvar(latents)
        return latents

    def decode(self, z):
        return self.decoder(z).permute(0, 2, 1)

    def forward(self, x):
        latents = self.encode(x)
        return self.decode(latents)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=5e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def context_fid(y_syn, y_real, model):
    model.eval()
    with torch.no_grad():
        y_syn_latent = model.encode(y_syn.to(model.device)).cpu().numpy()
        y_real_latent = model.encode(y_real.to(model.device)).cpu().numpy()
    assert len(y_real_latent.shape) == 2

    mu_real = np.mean(y_real_latent, axis=0)
    sigma_real = np.cov(y_real_latent, rowvar=False)

    mu_syn = np.mean(y_syn_latent, axis=0)
    sigma_syn = np.cov(y_syn_latent, rowvar=False)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_syn, sigma_syn)
    return fid


def get_encoder(train_dl, device):
    # train_dl = TensorDataset(y_real)
    # train_dl = DataLoader(train_dl, batch_size=64, shuffle=True)
    batch = next(iter(train_dl))

    model = VanillaAE(
        seq_len=batch["x"].shape[1], seq_dim=batch["x"].shape[-1], latent_dim=256
    )
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    for i in tqdm.tqdm(range(50)):
        # train_loss_epoch = 0
        for x in train_dl:
            x = batch["x"].to(device)
            optim.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), x)
            loss.backward()
            optim.step()

    return model


def ablation_metrics(y_syn, y_real, model):
    cfid = context_fid(y_syn, y_real, model)
    # sw = WassersteinDistances(y_real.squeeze(-1).numpy(), y_syn.squeeze(-1).numpy(), seed=9)
    return cfid
    # return cfid, sw.sliced_distances(num_directions=1000).mean()
