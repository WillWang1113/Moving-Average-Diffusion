import os

import pandas as pd

from src.utils.train import setup_seed

# root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings"
root_pth = "/home/user/data/FrequencyDiffusion/savings"
setup_seed()


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
        # "ECL": "electricity",
        # "ETTh1": "etth1",
        # "ETTh2": "etth2",
        # "ETTm1": "ettm1",
        # "ETTm2": "ettm2",
        # "Exchange": "exchange_rate",
        # "TrafficL": "traffic",
        # "Weather": "weather",
        "MFRED":"mfred"
    }

    model_name = "MADtime_pl_doublenorm"
    # pred_len = [96]
    # pred_len = [96, 192]
    # pred_len = [96, 192, 336]
    # pred_len = [96, 192, 336, 720]
    pred_len = [288,432,576]
    # save_dir = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings/"
    save_dir = root_pth
    all_df = []
    for d in ds:
        real_d = ds[d]
        ds_df = []
        for pl in pred_len:
            result_path = os.path.join(save_dir, f"{real_d}_{pl}_S",
                                       model_name, "dtm_.csv")
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
    # all_df.to_csv(f'{model_name}.csv')
    # all_df.to_csv(os.path.join(save_dir, 'result.csv'))
    # print(all_df.to_latex(float_format="{:.3f}".format))

    # import pandas as pd
    # df = pd.read_csv("assets/result.csv", index_col=[0,1], header=[0,1])
    # print(df.to_latex(float_format="{:.4f}".format))