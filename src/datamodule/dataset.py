import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datamodule.dataclass import TimeSeries

root_pth = "/home/user/data/FrequencyDiffusion"


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
    }
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    train_ds = TimeSeries(df_train, **CONFIG)
    test_ds = TimeSeries(df_test, **CONFIG)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


def mfred():
    df = pd.read_csv(
        os.path.join(root_pth, "MFRED_clean.csv"), index_col=0, parse_dates=True
    )
    dummy_ = pd.get_dummies(df[["hour", "weekday"]], prefix_sep=":").astype("float")
    df = pd.concat([df, dummy_], axis=1)
    CONFIG = yaml.safe_load(open("configs/mfred.yaml", "r"))
    data_config = CONFIG['data_config']
    data_config['freq_kw'] = CONFIG['diff_config']['freq_kw']
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    train_ds = TimeSeries(df_train, **data_config)
    test_ds = TimeSeries(df_test, **data_config)
    train_dl = DataLoader(train_ds, batch_size=data_config['batch_size'], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=data_config['batch_size'], shuffle=False)
    return train_dl, test_dl

def nrel():
    df = pd.read_csv(
        os.path.join(root_pth, "nrel_clean.csv"), index_col=0, parse_dates=True
    )
    CONFIG = yaml.safe_load(open("configs/nrel.yaml", "r"))
    data_config = CONFIG['data_config']
    data_config['freq_kw'] = CONFIG['diff_config']['freq_kw']
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    train_ds = TimeSeries(df_train, **data_config)
    test_ds = TimeSeries(df_test, **data_config)
    train_dl = DataLoader(train_ds, batch_size=data_config['batch_size'], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=data_config['batch_size'], shuffle=False)
    return train_dl, test_dl
    