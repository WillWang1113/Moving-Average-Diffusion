import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import torch
from torch.nn import AvgPool1d, functional
import matplotlib.pyplot as plt
from neuralforecast.losses.numpy import mae, mqloss, mse
from typing import Optional, Union
import ot
import tqdm
from src.layers.Autoformer_EncDec import series_decomp
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
import lightning as L

import copy
from torchvision.ops import MLP
import torch.nn.functional as F
from scipy import linalg

class WassersteinDistances:
    """Calculate Wasserstein distance of two datasets in various ways.

    Parameters
    ----------
    original_data : np.ndarray
        Original data set, an (n, d) ndarray.
    other_data : np.ndarray
        Other data set, which might be imputed or simulated data, also
        an (n, d) ndarray.
    normalisation : Normalisation
        Method of normalising data.  If 'none', no normalisation will be used.
        If 'standatdise', then standardise the data by dividing by the
        standard deviation of the original data.  (There is no need to
        subtract the mean, as this does not affect the Wasserstein distance.)

    """

    def __init__(
        self,
        original_data: np.ndarray,
        other_data: np.ndarray,
        normalisation: Optional[str] = "none",
        seed: Optional[int] = None,
    ) -> None:
        self.original_data = original_data
        self.other_data = other_data
        self.normalisation = normalisation
        self.rng = np.random.default_rng(seed)

    def random_direction(self, dim: int) -> np.ndarray:
        """Generate a unit vector in a random direction.

        Parameters
        ----------
        dim : int
            Dimension of vector to be generated.

        Returns
        -------
        unit_vector : np.ndarray
            A unit vector of shape (dim,).

        """
        vector = self.rng.normal(size=dim)
        vector_magnitude = np.linalg.norm(vector)
        unit_vector = vector / vector_magnitude
        return unit_vector

    def get_random_directions(self, n_directions: int):
        """Get random directions for an experiment.

        Parameters
        ----------
        n_directions : int
            The number of directions to produce.

        Returns
        -------
        directions : list[np.ndarray]
            A list of unit vectors specifying the directions to use.  The
            results will be given in the same order.

        """
        dimension = self.original_data.shape[1]
        directions = [self.random_direction(dimension) for _ in range(n_directions)]
        return directions

    def get_marginal_directions(self):
        """Get marginal directions for an experiment.

        These are just the standard basis vectors.

        Returns
        -------
        directions : list[np.ndarray]
            A list of standard unit vectors.

        """
        dimension = self.original_data.shape[1]
        directions = [np.identity(dimension)[i] for i in range(dimension)]
        return directions

    def feature_distance(self, feature: int):
        """Calculate the dataset distance for a specific feature.

        This calculates the Wasserstein 2-distance between the
        specified feature in the two datasets.

        Parameters
        ----------
        feature : int
            The column number of the feature to consider: 0, 1, 2, ...,
            `num_fields` - 1.

        Returns
        -------
        distance : float
            The Wasserstein 2-distance.

        """
        original = self.original_data[:, feature]
        other = self.other_data[:, feature]
        original_normalised, other_normalised = self._normalise(original, other)
        distance = ot.emd2_1d(original_normalised, other_normalised)
        distance = np.sqrt(distance)
        return float(distance)

    def directional_distance(self, direction: np.ndarray):
        """Calculate the dataset distance in a specified direction.

        This projects the two datasets onto the specified direction (that is,
        a 1-dimensional subspace), and calculates the Wasserstein distance
        between the two resulting distributions.

        Parameters
        ----------
        direction : np.array
            The direction in which to calculate the W_2 distance between
            the datasets.

        Returns
        -------
        distance : float
            The calculated W_2^2 distance.

        """
        original = self._project(self.original_data, direction)
        other = self._project(self.other_data, direction)
        original_normed, other_normed = self._normalise(original, other)
        distance = ot.emd2_1d(original_normed, other_normed)
        distance = np.sqrt(distance)
        return float(distance)

    @staticmethod
    def _project(data: np.ndarray, direction: np.ndarray) -> np.ndarray:
        proj = data @ direction
        assert isinstance(proj, np.ndarray)
        return proj

    def _normalise(
        self, orig: np.ndarray, other: np.ndarray
    ):
        if self.normalisation == "none":
            return orig, other
        if self.normalisation == "standardise":
            sd = np.std(orig)
            return orig / sd, other / sd
        raise ValueError(f"Unrecognised normalisation type: {self.normalisation}")

    def sliced_distances(self, num_directions: int):
        """Calculate the sliced Wasserstein distance between datasets.

        Args:
            num_directions (int): Number of directions in the sliced Wasserstein estimation.

        Returns:
            np.ndarray: distribution of Wasserstein distances over all directions.
        """
        directions = self.get_random_directions(num_directions)
        distances = []
        for direction in tqdm.tqdm(
            directions,
            desc="Computing sliced Wasserstein",
            unit="proj",
            leave=False,
            colour="blue",
        ):
            distances.append(self.directional_distance(direction))
        return np.array(distances)

    def marginal_distances(self) -> np.ndarray:
        """Calculate the marginal Wasserstein distances between datasets.

        Returns:
            np.ndarray: distribution of Wasserstein distances over all features.
        """
        n_features = self.original_data.shape[1]
        distances = []
        for feature in tqdm.tqdm(
            range(n_features),
            desc="Computing marginal Wasserstein",
            unit="feature",
            leave=False,
            colour="blue",
        ):
            distances.append(self.feature_distance(feature))
        return np.array(distances)
    
    

# def mqloss(
#     y: np.ndarray,
#     y_hat: np.ndarray,
#     quantiles: np.ndarray,
#     weights: Optional[np.ndarray] = None,
#     axis: Optional[int] = None,
# ) -> Union[float, np.ndarray]:
#     if weights is None:
#         weights = np.ones(y.shape)

#     # _metric_protections(y, y_hat, weights)
#     n_q = len(quantiles)

#     y_rep = np.expand_dims(y, axis=-1)
#     error = y_hat - y_rep
#     sq = np.maximum(-error, np.zeros_like(error))
#     s1_q = np.maximum(error, np.zeros_like(error))
#     mqloss = quantiles * sq + (1 - quantiles) * s1_q

#     # Match y/weights dimensions and compute weighted average
#     weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
#     mqloss = np.average(mqloss, weights=weights, axis=axis)

#     return mqloss


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    # mae = MAE(pred, true)
    # mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    point_pred = np.median(pred, axis=-1)
    mean_mse = MSE(np.mean(point_pred, axis=1), np.mean(true, axis=1))
    mse = MSE(point_pred, true)
    crps = mqloss(true, pred, quantiles=(np.arange(9) + 1) / 10)

    return mean_mse, mse, crps
    # return mae, mse, rmse, mape, mspe


# def crps(y_pred, y_real, sample_weight=None):
#     num_samples = y_pred.shape[0]
#     absolute_error = np.mean(np.abs(y_pred - y_real), axis=0)

#     if num_samples == 1:
#         return np.average(absolute_error, weights=sample_weight)

#     y_pred_sort = np.sort(y_pred, axis=0)
#     b0 = y_pred.mean(axis=0)
#     b1_values = y_pred_sort * np.arange(num_samples).reshape(
#         -1, *[1 for _ in range(len(y_pred.shape) - 1)]
#     )
#     b1 = b1_values.mean(axis=0) / num_samples

#     per_obs_crps = absolute_error + b0 - 2 * b1
#     return np.average(per_obs_crps, weights=sample_weight)


# def mpbl(y_pred, y_real, quantiles=[0.05 * (1 + i) for i in range(19)]):
#     y_pred_q = np.quantile(y_pred, quantiles, axis=0)
#     quantiles = np.array(quantiles).reshape(-1, *[1 for _ in range(y_pred_q.ndim - 1)])
#     error = y_real - y_pred_q  # * n*9
#     first_term = quantiles * error
#     second_term = (quantiles - 1) * error
#     loss = np.maximum(first_term, second_term)
#     return loss.mean()


# def mae(y_pred, y_real, normalize=False):
#     if y_pred.ndim <= 3:
#         y_pred_point = y_pred
#     elif y_pred.ndim == 4:
#         y_pred_point = np.median(y_pred, axis=0)
#     else:
#         raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
#     # y_pred_point = np.median(y_pred, axis=0)
#     scale = np.mean(np.abs(y_real)) if normalize else 1
#     return np.mean(np.abs(y_pred_point - y_real)) / scale


# def mse(y_pred, y_real):
#     if y_pred.ndim <= 3:
#         y_pred_point = y_pred
#     elif y_pred.ndim == 4:
#         y_pred_point = np.median(y_pred, axis=0)
#     else:
#         raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
#     # y_pred_point = np.median(y_pred, axis=0)
#     return np.mean((y_pred_point - y_real) ** 2)


# def rmse(y_pred, y_real, normalize=False):
#     if y_pred.ndim <= 3:
#         y_pred_point = y_pred
#     elif y_pred.ndim == 4:
#         y_pred_point = np.median(y_pred, axis=0)
#     else:
#         raise ValueError('wrong y_pred shape, either ndim ==4 or 3 less.')
#     MSE = np.mean((y_pred_point - y_real) ** 2)
#     scale = np.mean(np.abs(y_real)) if normalize else 1
#     return np.sqrt(MSE) / scale


def calculate_metrics(
    y_pred,
    y_real,
    quantiles=(np.arange(9) + 1) / 10,
    # normalize=False
):
    y_pred = y_pred.cpu().numpy()
    y_real = y_real.cpu().numpy()
    # print(y_pred.shape)
    # print(y_real.shape)

    # if isinstance(y_pred, np.ndarray) and isinstance(y_real, np.ndarray):
    print("Evaluate on highest resolution")
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

    # elif isinstance(y_pred, list) and isinstance(y_real, list):
    #     print("Evaluate on multi resolutions")
    #     all_res_metric = []
    #     all_res_pred_q = []
    #     all_res_pred_point = []

    #     for i in range(len(y_pred)):
    #         res_y_pred, res_y_real = y_pred[i], y_real[i]

    #         assert res_y_pred.ndim == 4
    #         assert res_y_pred[0].shape == y_real.shape
    #         res_y_pred_point = np.median(res_y_pred, axis=0)
    #         res_y_pred_q = np.quantile(res_y_pred, quantiles, axis=0)
    #         res_y_pred_q = np.transpose(res_y_pred_q, (1, 2, 3, 0))

    #         MAE = mae(res_y_real, res_y_pred_point)
    #         MSE = mse(res_y_real, res_y_pred_point)
    #         MQL = mqloss(res_y_real, res_y_pred_q, quantiles=np.array(quantiles))

    #         all_res_metric.append((MAE, MSE, MQL))
    #         all_res_pred_q.append(res_y_pred_q)
    #         all_res_pred_point.append(all_res_pred_point)

    #         # RMSE = rmse(res_y_pred, res_y_real, normalize)
    #         # MAE = mae(res_y_pred, res_y_real, normalize)
    #         # CRPS = crps(res_y_pred, res_y_real)
    #         # MPBL = mpbl(res_y_pred, res_y_real)
    #         # print(RMSE)
    #         # all_res_metric.append((RMSE, MAE, CRPS, MPBL))
    #     return all_res_metric, all_res_pred_q, all_res_pred_point
    # else:
    #     raise ValueError("wrong y_pred shape")


# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/pdf/2205.13504.pdf
#     """

#     def __init__(self, seq_len, pred_len, enc_in, moving_avg=25, individual=False):
#         """
#         individual: Bool, whether shared model among different variates.
#         """
#         super(Model, self).__init__()
#         self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)

#     def encoder(self, x):
#         x = self.Linear_Seasonal(x.permute(0,2,1))
#         x = x.permute(0, 2, 1)
#         return x

#     def forward(self, x_enc: torch.Tensor):
#         # Encoder
#         x = self.encoder(x_enc)
#         return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, seq_len, pred_len, enc_in, moving_avg=25, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.seq_len = seq_len

        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_in
        self.pred_len = pred_len

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x_enc: torch.Tensor):
        # Encoder
        x = self.encoder(x_enc)
        return x


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


def tstr_dlinear(y_syn, y_real, seq_len=96, pred_len=16, device="cpu"):
    y_syn = y_syn[:, : seq_len + pred_len, :]
    y_real = y_real[:, : seq_len + pred_len, :]
    # n_data = len(y_syn)
    x_train = y_syn[:, :seq_len]
    y_train = y_syn[:, seq_len:]
    x_test = y_real[:, :seq_len]
    y_test = y_real[:, seq_len:]

    train_dl = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dl, batch_size=64, shuffle=True)

    test_dl = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_dl, batch_size=256, shuffle=False)
    # test_dl = TensorDataset(x_test, y_test)
    model = Model(seq_len, pred_len, enc_in=y_syn.shape[-1]).to(device)
    optim = torch.optim.Adam(model.parameters())
    for i in tqdm.tqdm(range(50)):
        # train_loss_epoch = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optim.step()
            # train_loss_epoch = train_loss_epoch + loss
        # train_loss_epoch /= len(train_dl)
        # if (i + 1) % 10 == 0:
        #     print(i, train_loss_epoch)

    y_pred_tstr = (
        torch.concat([model(x.to(device)) for (x, y) in test_dl]).detach().cpu().numpy()
    )
    return MSE(y_pred_tstr, y_test.cpu().numpy())


# def tstr(y_pred, y_real):
#     if isinstance(y_pred, torch.Tensor):
#         y_pred = y_pred.cpu().numpy()
#     if isinstance(y_real, torch.Tensor):
#         y_real = y_real.cpu().numpy()
#     all_mse = []
#     # model = MLP
#     for i in range(y_pred.shape[-1]):
#         x, y, y_r = y_pred[:, -33:-1, i], y_pred[:, -1, [i]], y_real[:, -1, [i]]
#         x_train, x_test, y_train, _, _, y_test = train_test_split(
#             x, y, y_r, test_size=0.25, shuffle=False
#         )

#         reg_model = LinearRegression()
#         reg_model.fit(x_train, y_train)
#         tstr_y_pred = reg_model.predict(x_test)
#         all_mse.append(MSE(tstr_y_pred, y_test))

#     return np.mean(all_mse)


def sr_metrics(y_pred, y_real, kernel_size, autoencoder):
    y_pred = y_pred.squeeze(0)
    con_err = consistency_error(y_pred, y_real, kernel_size)
    lsd = log_spectral_distance(y_pred, y_real)
    mse = MSE(y_pred.cpu().numpy(), y_real.cpu().numpy())
    cfid = context_fid(y_pred, y_real, autoencoder)
    # st_lps = tstr_dlinear(y_pred, y_real, seq_len=96, pred_len=16)
    # lt_lps = tstr_dlinear(y_pred, y_real, seq_len=96, pred_len=192)
    # lps_ideal = tstr_dlinear(y_real, y_real)
    return mse, lsd, con_err, cfid
    # return mse


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

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

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
            x = batch['x'].to(device)
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