import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.datamodule import dataset
from src.models import diffusion
from src.utils.parser import exp_parser
from src.utils.train import Trainer, get_expname_, setup_seed
import json

# root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings"
root_pth = "/home/user/data/FrequencyDiffusion/savings"
setup_seed()


def main(args, n):
    device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)

    data_fn = getattr(dataset, args["dataset"])

    config = exp_parser(args)

    bb_, cn_, df_ = (
        config["bb_config"].get("name"),
        config["cn_config"].get("name"),
        config["diff_config"].pop("name"),
    )
    train_dl, val_dl, test_dl, scaler = data_fn()

    df = getattr(diffusion, df_)
    exp_name = (
        f"{get_expname_(config, bb_, cn_, df_)}{'t' if args['smoke_test'] else n}"
    )
    # exp_name = f"{bb_}_{CONFIG['diff_config']}_{'t' if args.test else n}"
    save_folder = os.path.join(
        root_pth,
        args["dataset"],
        df_,
        exp_name,
    )
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(config, w, indent=2)

    with open(os.path.join(root_pth, args["dataset"], "scaler.npy"),
              "wb") as f:
        np.save(f, scaler)
    torch.save(test_dl, os.path.join(root_pth, args["dataset"], "test_dl.pt"))
    batch = next(iter(train_dl))
    seq_length, seq_channels = (
        batch["observed_data"].shape[1],
        batch["observed_data"].shape[2],
    )
    target_seq_length, target_seq_channels = (
        batch["future_data"].shape[1],
        batch["future_data"].shape[2],
    )
    future_seq_length, future_seq_channels = (
        batch["future_features"].shape[1],
        batch["future_features"].shape[2],
    )

    config["bb_config"]["seq_channels"] = target_seq_channels
    config["bb_config"]["seq_length"] = target_seq_length
    config["bb_config"]["latent_dim"] = config["cn_config"]["latent_dim"]
    config["cn_config"]["seq_channels"] = seq_channels
    config["cn_config"]["seq_length"] = seq_length
    config["cn_config"]["future_seq_channels"] = future_seq_channels
    config["cn_config"]["future_seq_length"] = future_seq_length

    diff = df(
        backbone_config=config["bb_config"],
        conditioner_config=config["cn_config"],
        **config["diff_config"],
    )

    trainer = Trainer(
        smoke_test=args["smoke_test"],
        device=device,
        output_pth=save_folder,
        exp_name=exp_name,
        **config["train_config"],
    )

    trainer.fit(diff, train_dl, val_dl)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Hyperparameters config")
    # parser.add_argument(
    #     "-c", "--config", required=True, help="Path to the YAML configuration file."
    # )
    # parser.add_argument("--dataset", type=str, default="mfred")
    # parser.add_argument("--smoke_test", action="store_true")
    # parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--num_train", type=int, default=5)
    # # Define overrides with dot notation
    # parser.add_argument("--diff_config.name", type=str)
    # parser.add_argument("--diff_config.norm", action="store_true", default=None)
    # parser.add_argument("--diff_config.pred_diff", action="store_true", default=None)
    # parser.add_argument("--diff_config.noise_kw.name", type=str)
    # parser.add_argument("--diff_config.noise_kw.min_beta", type=float)
    # parser.add_argument("--diff_config.noise_kw.max_beta", type=float)
    # parser.add_argument("--bb_config.name", type=str)
    # args = parser.parse_args()
    # for i in range(args.num_train):
    #     # args = vars(args)
    #     main(vars(args), i)

    # # SAVE METRICS
    # ds = [
    #     "ETTh1",
    #     "ETTh2",
    #     "ETTm1",
    #     "ETTm2",
    #     "ECL",
    #     "Exchange",
    #     "TrafficL",
    #     "Weather",
    # ]
    # pred_len = [96, 192, 336, 720]
    # save_dir = "/mnt/ExtraDisk/wcx/research/benchmarks"
    # all_df = []
    # for d in ds:
    #     ds_df = []
    #     for pl in pred_len:
    #         result_path = os.path.join(save_dir, f"{d}_96_{pl}_U","results.csv")
    #         df = pd.read_csv(result_path, index_col=0)
    #         df.index.name = 'model'
    #         df = df.reset_index()
    #         df = df.groupby('model').mean()
    #         df = df.drop(columns=['MAE', 'iter'])
    #         df = df.stack()
    #         df = pd.DataFrame(df, columns=[pl]).transpose()
    #         ds_df.append(df)
    #     ds_df = pd.concat(ds_df)
    #     ds_df.index.name = 'pred_len'
    #     ds_df['dataset'] = d
    #     ds_df = ds_df.reset_index()
    #     ds_df = ds_df.set_index(['dataset','pred_len'])
    #     all_df.append(ds_df)
    # all_df = pd.concat(all_df)
    # all_df.to_csv(os.path.join(save_dir, 'bench_result.csv'))
    # # print(all_df.to_latex(float_format="{:.3f}".format))

    # SAVE METRICS
    ds = {
        "ETTh1": "etth1",
        "ETTh2": "etth2",
        "ETTm1": "ettm1",
        "ETTm2": "ettm2",
        "ECL": "electricity",
        "Exchange": "exchange_rate",
        "TrafficL": "traffic",
        "Weather": "weather",
    }

    model_name = "MADfreq_v2"
    pred_len = [96, 192, 336, 720]
    # save_dir = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings/"
    save_dir = root_pth
    all_df = []
    for d in ds:
        real_d = ds[d]
        ds_df = []
        for pl in pred_len:
            result_path = os.path.join(save_dir, f"{real_d}_{pl}_S",
                                       model_name, "fast_dtm_.csv")
            df = pd.read_csv(result_path, index_col=0)

            df = df.drop(columns=["MAE", "granularity"])
            df["method"] = model_name
            df = df.set_index("method")
            df = df.stack()
            df = pd.DataFrame(df, columns=[pl]).transpose()
            ds_df.append(df)

        ds_df = pd.concat(ds_df)
        ds_df.index.name = 'pred_len'
        ds_df['dataset'] = d
        ds_df = ds_df.reset_index()
        ds_df = ds_df.set_index(['dataset', 'pred_len'])
        all_df.append(ds_df)
    all_df = pd.concat(all_df)
    print(all_df)
    # all_df.to_csv('')

    # all_bench_df = pd.read_csv(
    #     '/mnt/ExtraDisk/wcx/research/benchmarks/bench_result.csv',
    #     index_col=[0, 1],
    #     header=[0, 1])
    # all_df = pd.concat([all_df, all_bench_df], axis=1)
    # print(all_df)
    all_df.to_csv(os.path.join(save_dir, 'result.csv'))
    # print(all_df.to_latex(float_format="{:.3f}".format))

    # import pandas as pd
    # df = pd.read_csv("assets/result.csv", index_col=[0,1], header=[0,1])
    # print(df.to_latex(float_format="{:.4f}".format))