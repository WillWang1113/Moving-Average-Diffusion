import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = int((configs.seq_len - configs.ma_ks) / configs.ma_stride + 1)
        # self.seq_len = configs.seq_len
        # self.pred_len = int((configs.pred_len - configs.ma_ks) / configs.ma_stride + 1)
        if (
            self.task_name == "classification"
            or self.task_name == "anomaly_detection"
            or self.task_name == "imputation"
        ):
            self.pred_len = configs.seq_len
        else:
            self.pred_len = int(
                (configs.pred_len - configs.ma_ks) / configs.ma_stride + 1
            )

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.n_quantile = configs.n_quantile
        self.configs = configs

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
        if configs.prob == 1:
            self.quantile_linear = nn.Linear(
                self.pred_len, self.n_quantile * self.pred_len
            )

        if configs.pred_stats == 1:
            self.mean_linear = nn.Linear(self.pred_len, 1)
            self.std_linear = nn.Linear(self.pred_len, 1)

        if self.task_name == "classification":
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class
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

    def forecast(self, x_enc: torch.Tensor):
        x_enc = torch.nn.functional.avg_pool1d(
            x_enc.permute(0, 2, 1),
            self.configs.ma_ks,
            stride=self.configs.ma_stride,
        ).permute(0, 2, 1)

        # Encoder
        x = self.encoder(x_enc)
        if self.configs.prob == 1:
            x = self.quantile_linear(x.permute(0, 2, 1)).reshape(
                -1, self.channels, self.pred_len, self.n_quantile
            )
            x = x.permute(0, 2, 1, 3)

        if self.configs.pred_stats == 1:
            x_mean = self.mean_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            x_std = self.std_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = torch.concat([x_mean, x_std], dim=1)

        # return x.permute(0,2,1,3)
        return x

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
