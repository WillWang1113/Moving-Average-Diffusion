import logging
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT, Autoformer, FEDformer, LSTM, MLP, NBEATSx, DeepAR
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.losses.numpy import mqloss
from src.utils.metrics import calculate_metrics, get_bench_metrics

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


df = pd.read_csv(
    "/home/user/data/FrequencyDiffusion/dataset/MFRED_clean.csv",
    index_col=0,
    parse_dates=True,
)
# df['ds'] = df.index
df = df.drop(columns=["weekday", "hour"])
df["unique_id"] = 1.0
df = df.reset_index()
n_series = len(df.unique_id.unique())

quantiles = [0.05 * (q + 1) for q in range(19)]
horizon = 288
config = {
    "h": horizon,
    "input_size": horizon,
    "max_steps": 300,
    # "loss": MQLoss(quantiles=quantiles),
    "loss": DistributionLoss(quantiles=quantiles),
    "futr_exog_list": ["t2m"],
    "early_stop_patience_steps": 5,
    "val_check_steps": 10,
    "inference_windows_batch_size": 1024,
    "windows_batch_size": 512,
    "scaler_type":'standard'
}

models = [
    # MLP(**config),
    # NHITS(**config),
    # NBEATSx(**config),
    # TFT(**config),
    # Autoformer(**config),
    # FEDformer(**config),
    DeepAR(**config)
]
model_names = [m._get_name() for m in models]
nf = NeuralForecast(models=models, freq="5min")

Y_hat_df = nf.cross_validation(
    df=df,
    val_size=int(len(df) * 0.8 / 7),
    test_size=21599 - 288,
    target_col="value",
    n_windows=None,
)
# Y_hat_df.to_csv('/home/user/data/FrequencyDiffusion/savings/mfred/benchmarks.csv')
cols = Y_hat_df.columns.tolist()
y_true = Y_hat_df.value.values.reshape(-1, horizon, n_series)

for i, m in enumerate(model_names):
    model_cols = [c for c in cols if m in c]

    y_hat = Y_hat_df[model_cols].values
    y_hat = y_hat.reshape(-1, horizon, n_series, len(quantiles))
    # y_hat = y_hat.sort(axis=-1)

    (RMSE, MAE, PBL) = get_bench_metrics(y_hat, y_true, quantiles=np.array(quantiles))

    print(m)
    print((RMSE, MAE, PBL))
    # fig, ax = plt.subplots()
    # ax.plot(y_true[127])
    # ax.plot(y_hat[127].squeeze())
    # fig.savefig("test.png")


# y_true = y_true.reshape(n_series, -1, horizon).transpose(1, 2, 0)
# y_hat = y_hat.reshape(n_series, -1, horizon).transpose(1, 2, 0)

# print("\n" * 4)
# print("Parsed results")
# print("y_true.shape (n_series, n_windows, n_time_out):\t", y_true.shape)
# print("y_hat.shape  (n_series, n_windows, n_time_out):\t", y_hat.shape)

# # # print('MSE: ', mse(y_hat, y_true))
# print("MAE: ", mae(y_hat, y_true))
