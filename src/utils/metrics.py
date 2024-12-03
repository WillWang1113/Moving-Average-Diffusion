import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import torch
from torch.nn import AvgPool1d, functional
import matplotlib.pyplot as plt
from neuralforecast.losses.numpy import mae, mqloss, mse
from typing import Optional, Union
from src.layers.Autoformer_EncDec import series_decomp
from torch.utils.data import DataLoader, TensorDataset

from torch import nn


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



def tstr_dlinear(y_pred, y_real, seq_len=96, pred_len=16):

    y_pred = y_pred[:, :seq_len+pred_len, :]
    y_real = y_real[:, :seq_len+pred_len, :]
    n_data = len(y_pred)
    x_train = y_pred[:int(0.8*n_data), :seq_len]
    y_train = y_pred[:int(0.8*n_data), seq_len:]
    x_test = y_real[int(0.8*n_data):, :seq_len]
    y_test = y_real[int(0.8*n_data):, seq_len:]

    train_dl = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dl, batch_size=64)
    
    test_dl = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_dl, batch_size=256, shuffle=False)
    # test_dl = TensorDataset(x_test, y_test)
    model = Model(seq_len, pred_len, enc_in=y_pred.shape[-1]).to(y_real.device)
    optim = torch.optim.Adam(model.parameters())
    for i in range(50):
        train_loss_epoch=0
        for (x, y) in train_dl:
            optim.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optim.step()
            train_loss_epoch=train_loss_epoch+loss
        train_loss_epoch/=len(train_dl)
        if (i+1) % 10 ==0:
            print(i, train_loss_epoch)
    
    y_pred_tstr = torch.concat([model(x) for (x, y) in test_dl]).detach().cpu().numpy()
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


def sr_metrics(y_pred, y_real, kernel_size):
    y_pred = y_pred.squeeze(0)
    con_err = consistency_error(y_pred, y_real, kernel_size)
    lsd = log_spectral_distance(y_pred, y_real)
    mse = MSE(y_pred.cpu().numpy(), y_real.cpu().numpy())
    st_lps = tstr_dlinear(y_pred, y_real, seq_len=96, pred_len=16)
    lt_lps = tstr_dlinear(y_pred, y_real, seq_len=96, pred_len=192)
    # lps_ideal = tstr_dlinear(y_real, y_real)
    return (mse, lsd, con_err, st_lps, lt_lps)
    # return mse
