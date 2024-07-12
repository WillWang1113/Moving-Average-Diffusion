import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, mqloss, mse
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import (
    NHITS,
    # iTransformer,
    # TFT,
    FEDformer,
    DLinear,
    PatchTST,
    TimesNet,
)

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def main(args):
    Y_df, _, _ = LongHorizon.load(directory='/home/user/data/NF_longterm', group=args['dataset'])
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    if args["task"] == "U":
        Y_df = Y_df[Y_df['unique_id'] == 'OT']
    
    n_series = LongHorizonInfo[args['dataset']].n_ts

    if args["dataset"].__contains__("ETTh"):
        num_train = 12 * 30 * 24
        num_test = 4 * 30 * 24
        num_vali = 4 * 30 * 24
    elif args["dataset"].__contains__("ETTm"):
        num_train = 12 * 30 * 24 * 4
        num_test = 4 * 30 * 24 * 4
        num_vali = 4 * 30 * 24 * 4
    else:
        num_train = int(len(Y_df) * 0.7)
        num_test = int(len(Y_df) * 0.2)
        num_vali = len(Y_df) - num_train - num_test
    # quantiles = [0.025 * (1 + i) for i in range(39)]
    level = [i for i in range(10, 100, 10)]
    quantiles = [0.05 * (1 + i) for i in range(19)]
    seq_len = args["seq_len"]
    pred_len = args["pred_len"]
    config = {
        "h": pred_len,
        "input_size": seq_len,
        "max_steps": 100,
        "loss": MQLoss(level=level),
        "early_stop_patience_steps": 5,
        "val_check_steps": 1,
        "inference_windows_batch_size": 1024,
        "windows_batch_size": 128,
        "scaler_type": "standard",
        "fast_dev_run": args["fast_dev_run"],
        "num_sanity_val_steps": 0,
        # "devices":[0],
        "random_seed": args["n"],
        "enable_progress_bar": False,
        "default_root_dir": "/home/user/data/FrequencyDiffusion/savings/mfred/benchmarks",
    }

    models = [
        # TSMixer(**config, n_series=n_series, loss=MQLoss(level=level)),
        PatchTST(
            hidden_size=256,
            dropout=0.1,
            head_dropout=0.1,
            fc_dropout=0.1,
            **config,
        ),
        TimesNet(**config),
        DLinear(**config),
        NHITS(**config),
        FEDformer(**config),
        # FEDformer(**config, loss=MQLoss(level=level)),
        # DeepAR(
        #     **config,
        #     loss=DistributionLoss(distribution="Normal", num_samples=200, level=level),
        # ),
    ]
    model_names = [m._get_name() for m in models]
    nf = NeuralForecast(models=models, freq="5min")

    Y_hat_df = nf.cross_validation(
        df=Y_df,
        val_size=num_vali,
        test_size=num_test,
        n_windows=None,
    )
    cols = Y_hat_df.columns.tolist()
    y_true = Y_hat_df["y"].values.reshape(-1, pred_len, n_series)

    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(os.path.join(args["folder"], "true.npy")):
        np.save(os.path.join(args["folder"], "true.npy"), y_true)

    out_dict = {}
    for i, m in enumerate(model_names):
        print(m)
        model_cols = [c for c in cols if m + "-" in c]
        point_c = m + "-" + "median"
        y_point = Y_hat_df[point_c].values.reshape(-1, pred_len, n_series)

        y_hat = Y_hat_df[model_cols].values
        y_hat = y_hat.reshape(-1, pred_len, n_series, len(quantiles))

        MAE = mae(y_true, y_point)
        MSE = mse(y_true, y_point)
        MQL = mqloss(y_true, y_hat, quantiles=np.array(quantiles))

        out_dict[m] = (MAE, MSE, MQL)
        if not args["data_pth"].__contains__("MFRED"):
            np.save(os.path.join(args["folder"], f"pred_{m}_{args['n']}.npy"), y_hat)

    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", default=5, type=int)
    parser.add_argument("--task", default="U", type=str, choices=["M", "U"])
    parser.add_argument("--seq_len", default=96, type=int)
    parser.add_argument("--pred_len", default=96, type=int)
    parser.add_argument(
        "--dataset",
        default="ETTh1",
        type=str,
        choices=[
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "ECL",
            "Exchange",
            "Traffic",
            "Weather",
            "ILI",
        ],
    )
    # parser.add_argument(
    #     "--root_pth", default="/home/user/data/THU-timeseries", type=str
    # )
    # parser.add_argument("--data_pth", default="ETT-small/ETTh1_NF.csv", type=str)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    args = vars(args)
    folder = f"/home/user/data/FrequencyDiffusion/savings/{args['dataset']}_{args['seq_len']}_{args['pred_len']}_{args['task']}"
    args["folder"] = folder
    all_out_dict = []
    for i in range(args["num_train"]):
        args["n"] = i
        out_dict = main(args)
        all_out_dict.append(out_dict)

    with open(os.path.join(folder, "benchmarks.pkl"), "wb") as f:
        pickle.dump(all_out_dict, f)
