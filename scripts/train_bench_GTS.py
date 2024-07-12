import argparse
import os
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset, infer_freq
from gluonts.dataset.common import ListDataset
from gluonts.torch import SimpleFeedForwardEstimator, DLinearEstimator
from gluonts.evaluation import Evaluator
from gluonts.dataset.split import split
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import period_index
from pts.model.time_grad import TimeGradEstimator
from pts import Trainer
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper


def _to_dataframe(input_label) -> pd.DataFrame:
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
    return pd.DataFrame(full_target.transpose(), index=index)


def make_rolling_evaluation_predictions(
    dataset, predictor, num_samples: int = 100, num_windows=1, distance=1
):
    window_length = predictor.prediction_length + predictor.lead_time
    # print(predictor.lead_time)
    _, test_template = split(dataset, offset=0)
    test_data = test_template.generate_instances(
        window_length, windows=num_windows, distance=distance
    )
    return (
        predictor.predict(test_data.input, num_samples=num_samples),
        map(_to_dataframe, test_data),
    )


def main(args):
    device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)
    df = pd.read_csv(
        os.path.join(args["root_pth"], args["data_pth"]), index_col=0, parse_dates=True
    )
    if args["task"] == "U":
        df = df[["OT"]]
        target = "OT"
    else:
        target = df.columns.tolist()
    df = df.astype(float)
    print(df)
    freq = infer_freq(df.index)
    n_time = len(df)
    if args["data_pth"].__contains__("ETTh"):
        num_train = 12 * 30 * 24
        num_test = 4 * 30 * 24
        num_vali = 4 * 30 * 24
    elif args["data_pth"].__contains__("ETTm"):
        num_train = 12 * 30 * 24 * 4
        num_test = 4 * 30 * 24 * 4
        num_vali = 4 * 30 * 24 * 4
    else:
        num_train = int(n_time * 0.7)
        num_test = int(n_time * 0.2)
        num_vali = n_time - num_train - num_test

    # multiple_ts = {}
    # for c in df.columns.tolist():
    #     sub_df = df[[c]].copy()
    #     sub_df["item_id"] = c
    #     sub_df = sub_df.rename(columns={c: "target"})
    #     print(sub_df)
    #     multiple_ts[c] = sub_df
    print(str(df.index[0]))
    # print(df)
    for c in df.columns.tolist():
        print(df[:-num_test][c].values)
    train = ListDataset(
        [
            {"start": str(df.index[0]), "target": df[:-num_test][c].values}
            for c in df.columns.tolist()
        ],
        freq=freq
    )
    test = ListDataset(
        [{"start": str(df.index[0]), "target": df[c].values} for c in df.columns.tolist()],
        freq=freq
    )
    print(len(train))
    n_series = len(train)

    train_grouper = MultivariateGrouper(max_target_dim=n_series)
    for i in train:
        print(i)

    # test_grouper = MultivariateGrouper(
    #     num_test_dates=int(len(test) / len(train)), max_target_dim=min(2000, n_series)
    # )

    dataset_train = train_grouper(train)
    # dataset_test = test_grouper(test)

    # train = PandasDataset(
    #     {item_id: df[:num_train] for item_id, df in multiple_ts.items()}
    # )
    # valid = PandasDataset(
    #     {
    #         item_id: df[num_train : num_train + num_vali]
    #         for item_id, df in multiple_ts.items()
    #     }
    # )
    # test = PandasDataset(
    #     {item_id: df[num_train + num_vali :] for item_id, df in multiple_ts.items()},
    # )
    # print(len)
    # estimator = DLinearEstimator(
    #     prediction_length=args["pred_len"],
    #     context_length=args["seq_len"],
    #     trainer_kwargs={'accelerator':'cpu', 'max_steps':1}
    # )
    estimator = TimeGradEstimator(
        input_size=1600000,
        target_dim=n_series,
        prediction_length=args["pred_len"],
        context_length=args["seq_len"],
        cell_type="GRU",
        freq=freq,
        loss_type="l2",
        scaling=True,
        diff_steps=100,
        beta_end=0.1,
        beta_schedule="linear",
        trainer=Trainer(
            device=device,
            epochs=2,
            learning_rate=1e-3,
            num_batches_per_epoch=100,
            batch_size=64,
        )
    )
    predictor = estimator.train(dataset_train)

    # forecast_it, ts_it = make_rolling_evaluation_predictions(
    #     dataset=test, predictor=predictor, num_windows=2
    # )

    # forecasts_pytorch = list(forecast_it)
    # tss_pytorch = list(ts_it)

    # evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:], num_workers=0)
    # agg_metrics, item_metrics = evaluator(tss_pytorch, forecasts_pytorch)
    # print(item_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", default=5, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--task", default="U", type=str, choices=["M", "U"])
    parser.add_argument("--seq_len", default=96, type=int)
    parser.add_argument("--pred_len", default=96, type=int)
    parser.add_argument(
        "--root_pth", default="/home/user/data/THU-timeseries", type=str
    )
    parser.add_argument("--data_pth", default="ETT-small/ETTh1.csv", type=str)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    args = vars(args)
    folder = f"/home/user/data/FrequencyDiffusion/savings/gts_{args['data_pth'].split('/')[-1][:-7]}_{args['seq_len']}_{args['pred_len']}_{args['task']}"
    args["folder"] = folder
    all_out_dict = []
    for i in range(args["num_train"]):
        args["n"] = i
        out_dict = main(args)
        all_out_dict.append(out_dict)

    # with open(os.path.join(folder, "benchmarks.pkl"), "wb") as f:
    #     pickle.dump(all_out_dict, f)
