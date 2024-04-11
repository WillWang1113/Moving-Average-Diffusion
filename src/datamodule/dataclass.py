import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class TimeSeries(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        n_in: int,
        n_out: int,
        shift: int,
        col_in: list,
        col_out: list,
        tp_to_predict: torch.Tensor,
        observed_tp: torch.Tensor = None,
        **kwargs,
    ):
        cols = df.columns.to_list()
        window_in_features, window_out_features, window_target = [], [], []

        for c in cols:
            # print(c)
            if c.split(":")[0] in col_in:
                # print('sel_in')
                window_in_features.append(cols.index(c))
            if c.split(":")[0] in col_out:
                # print('sel_out')
                window_out_features.append(cols.index(c))
            if c.split(":")[0] == target:
                # print('target')
                window_target.append(cols.index(c))

        window_width = shift + n_out + n_in

        # Slide the total window
        windows = np.lib.stride_tricks.sliding_window_view(
            df.values, window_width, axis=0
        )
        windows = windows.transpose(0, 2, 1)

        hist_var = windows[:, :n_in, window_in_features]
        future_var = windows[:, -n_out:, window_out_features]
        # static_var = None
        fc_target = windows[:, -n_out:, window_target]

        print("observed data shape:\t", hist_var.shape)
        print("future features shape:\t", future_var.shape)
        print("forecast target shape:\t", fc_target.shape)

        self.his_data = torch.from_numpy(hist_var).float()
        self.future_features = torch.from_numpy(future_var).float()
        self.fc_data = torch.from_numpy(fc_target).float()
        self.tp_tp_predict = tp_to_predict.expand(len(hist_var), -1, len(window_target))
        self.observed_tp = (
            observed_tp.expand(len(hist_var), -1, len(window_in_features))
            if observed_tp is not None
            else None
        )

        # if kwargs.get('avg_list')

    def __len__(self):
        return len(self.fc_data)

    def __getitem__(self, index):
        batch_data = {
            "observed_data": self.his_data[index],
            "tp_to_predict": self.tp_tp_predict[index],
            "future_data": self.fc_data[index],
        }
        if self.observed_tp.numel():
            batch_data["observed_tp"] = self.observed_tp[index]
        if self.future_features.numel():
            batch_data["future_features"] = self.future_features[index]
        return batch_data

