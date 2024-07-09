import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datamodule.dataclass import TimeSeries

# root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/dataset"
root_pth = "/home/user/data/FrequencyDiffusion/dataset"


# toy dataset
def syntheic_sine(t_steps=4000, batch_size=128, freq_kw=None):
    ti = np.linspace(0, 20 * np.pi, t_steps)
    period_t_steps = int(t_steps / 10)

    def ft(t):
        # return torch.sin(t + x0)
        return np.sin(t) + np.sin(2 * t) + 0.5 * np.sin(12 * t)

    df = pd.DataFrame(
        np.concatenate(
            [
                ft(ti).reshape(-1, 1),
            ],
            axis=-1,
        ),
        columns=["value"],
    )
    # df = pd.DataFrame(ft(ti) + np.random.rand(*ti.shape)*0.1, columns=['value'])
    CONFIG = {
        "target": "value",
        "n_in": period_t_steps,
        "n_out": period_t_steps,
        "shift": 0,
        "col_in": ["value"],
        "col_out": [],
        "freq_kw": freq_kw,
        "cols": df.columns.tolist(),
    }

    window_width = CONFIG["shift"] + CONFIG["n_out"] + CONFIG["n_in"]

    # Slide the total window
    windows = np.lib.stride_tricks.sliding_window_view(df.values, window_width, axis=0)
    windows = windows.transpose(0, 2, 1)

    train_window, test_window = train_test_split(windows, test_size=0.2, shuffle=False)

    train_ds = TimeSeries(train_window, **CONFIG)
    test_ds = TimeSeries(test_window, **CONFIG)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


def mfred(data_config):
    df = pd.read_csv(
        os.path.join(root_pth, "MFRED_clean.csv"), index_col=0, parse_dates=True
    )
    df[["hour", "weekday"]] = df[["hour", "weekday"]].astype("category")
    dummy_ = pd.get_dummies(df[["hour", "weekday"]], prefix_sep=":").astype("float")
    df = pd.concat([df, dummy_], axis=1)
    df = df.drop(columns=["hour", "weekday"])
    # data_config = yaml.safe_load(open("configs/mfred.yaml", "r"))
    data_config["cols"] = df.columns.tolist()

    ct = ColumnTransformer(
        [("numbers", StandardScaler(), ["value", "t2m"])], remainder="passthrough"
    )

    train_df, _ = train_test_split(df, test_size=0.2, shuffle=False)
    if data_config["scale"]:
        ct = ct.fit(train_df)
        df = ct.transform(df)
        y_scaler = {
            "data": (
                ct.named_transformers_["numbers"].mean_,
                ct.named_transformers_["numbers"].scale_,
            ),
            "target": data_config["cols"].index(data_config["target"]),
        }
    else:
        y_scaler = None

    # Slide the total window
    window_width = data_config["shift"] + data_config["n_out"] + data_config["n_in"]
    windows = np.lib.stride_tricks.sliding_window_view(df, window_width, axis=0)
    windows = windows.transpose(0, 2, 1)

    train_window, test_window = train_test_split(
        windows, train_size=len(train_df) - window_width + 1, shuffle=False
    )

    train_window, val_window = train_test_split(
        train_window, test_size=1 / 7, shuffle=False
    )

    train_ds = TimeSeries(train_window, **data_config)
    val_ds = TimeSeries(val_window, **data_config)
    test_ds = TimeSeries(test_window, **data_config)
    train_dl = DataLoader(
        train_ds,
        batch_size=data_config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=data_config["batch_size"],
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=data_config["batch_size"],
        shuffle=False,
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl, y_scaler


