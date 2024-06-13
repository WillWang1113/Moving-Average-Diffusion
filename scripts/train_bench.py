import logging
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    NHITS,
    TFT,
    Autoformer,
    FEDformer,
    LSTM,
    MLP,
    NBEATSx,
    DeepAR,
)
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.losses.numpy import mqloss
from src.utils.metrics import calculate_metrics, get_bench_metrics
import pickle

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
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

    quantiles = [0.025 * (1 + i) for i in range(39)]
    level = [i for i in range(5, 100, 5)]
    # quantiles = [0.025 * (1 + i) for i in range(39)]
    horizon = args["horizon"]
    config = {
        "h": horizon,
        "input_size": horizon,
        "max_steps": 300,
        # "loss": DistributionLoss(
        #     distribution="Normal", num_samples=200,level=level
        # ),
        "valid_loss": MQLoss(level=level),
        "futr_exog_list": ["t2m"],
        "early_stop_patience_steps": 5,
        "val_check_steps": 10,
        # "inference_windows_batch_size": 2,
        # "windows_batch_size": 2,
        "inference_windows_batch_size": 1024,
        "windows_batch_size": 128,
        "scaler_type": "standard",
        # "fast_dev_run": True,
        "num_sanity_val_steps": 0,
        "random_seed": args["n"],
        "default_root_dir": "/home/user/data/FrequencyDiffusion/savings/mfred/benchmarks",
    }

    models = [
        MLP(**config, loss=MQLoss(level=level)),
        NHITS(**config, loss=MQLoss(level=level)),
        NBEATSx(**config, loss=MQLoss(level=level)),
        TFT(**config, loss=MQLoss(level=level)),
        Autoformer(**config, loss=MQLoss(level=level)),
        FEDformer(**config, loss=MQLoss(level=level)),
        # DeepAR(
        #     **config,
        #     loss=DistributionLoss(distribution="Normal", num_samples=200, level=level),
        # ),
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

    out_dict = {}
    for i, m in enumerate(model_names):
        print(m)
        model_cols = [c for c in cols if m in c]
        # print(Y_hat_df[model_cols])

        y_hat = Y_hat_df[model_cols].values
        y_hat = y_hat.reshape(-1, horizon, n_series, len(quantiles))
        # print(y_hat.shape)
        # print(y_true.shape)
        # y_hat = y_hat.sort(axis=-1)

        (RMSE, MAE, PBL) = get_bench_metrics(
            y_hat, y_true, quantiles=np.array(quantiles)
        )
        out_dict[m] = (RMSE, MAE, PBL)
        print((RMSE, MAE, PBL))
        # fig, ax = plt.subplots()
        # ax.plot(y_true[127])
        # ax.plot(y_hat[127].squeeze())
        # fig.savefig("test.png")

    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", default=5, type=int)
    parser.add_argument("--horizon", default=288, type=int)
    args = parser.parse_args()
    args = vars(args)
    all_out_dict = []
    for i in range(args["num_train"]):
        args["n"] = i
        out_dict = main(args)
        all_out_dict.append(out_dict)

    with open(
        "/home/user/data/FrequencyDiffusion/savings/mfred/benchmarks/bench_metrics.pkl",
        "wb",
    ) as f:
        pickle.dump(all_out_dict, f)
