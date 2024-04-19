import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from src.datamodule.dataclass import TimeSeries
from sklearn.model_selection import train_test_split

# toy dataset
def syntheic_sine(t_steps=2000, batch_size=128, freq_kw=None):
    ti = np.linspace(0, 20 * np.pi, t_steps)
    period_t_steps = int(t_steps / 10)
    tp_train = torch.linspace(0, 4 * np.pi, period_t_steps * 2)

    def ft(t):
        # return torch.sin(t + x0)
        return np.sin(t) + np.sin(2 * t) + 0.5 * np.sin(12 * t) + np.random.rand(*t.shape) * 0.1

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
        "observed_tp": tp_train[:period_t_steps].unsqueeze(0).unsqueeze(-1),
        "tp_to_predict": tp_train[period_t_steps:].unsqueeze(0).unsqueeze(-1),
        "freq_kw": freq_kw
    }
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    train_ds = TimeSeries(df_train, **CONFIG)
    test_ds = TimeSeries(df_test, **CONFIG)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl
