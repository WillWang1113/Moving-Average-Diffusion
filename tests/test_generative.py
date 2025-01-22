import os

import pandas as pd
import numpy as np

from src.utils.train import setup_seed

# root_pth = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings"
root_pth = "savings/MAD_fcst/savings"
# root_pth = "/home/user/data/MAD_fcst/savings"
# root_pth = "/home/user/data/FrequencyDiffusion/savings"
setup_seed()


if __name__ == "__main__":

    # SAVE METRICS
    ds = {
        "ECL": "electricity",
        # "ETTh1": "etth1",
        "ETTh2": "etth2",
        # "ETTm1": "ettm1",
        "ETTm2": "ettm2",
        "Exchange": "exchange_rate",
        "traffic": "traffic",
        "weather": "weather",
        # "MFRED":"mfred"
    }
    bs = 64
    # base_model_name = "MADtime_pl_Full_SETKS_FreqDoi"
    # base_model_name = "MADtime_pl_Full_SETKS"
    # base_model_name = "MADtime_learnmean"
    # base_model_name = "MADtime_FactOnly_SETKS_learnmean_freqdenoise_puncond0.5"
    # base_model_name = "MADtime_FactOnly_SETKS_learnmean_freqdenoise"
    # base_model_name = "MADtime_learnmean_freqdenoise"
    # base_model_name = "MADtime_pl_FactOnly_SETKS_FreqDoi_CFG_puncond0.8"
    # base_model_name = "MADtime_pl_FactOnly_FreqDoi"
    # base_model_name = "MADfreq"
    # base_model_name = "MADfreq_FactOnly"
    model_name = "MADTC_NFD_MLP_x0_bs64_condfcst"
    # base_model_name = "MADfreq_puncond0.5"
    # base_model_name = "MADfreq_FactOnly_puncond0.5"

    # model_name = f"MADtime_pl_Full_SETKS_bs{bs}"

    # model_name = f"MADtime_pl_FactOnly_FreqDoi_bs{bs}"
    # model_name = base_model_name + f'_bs{bs}'
    # model_name = "MADfreq_pl_doublenorm_zerodp"
    # pred_len = [96]
    # pred_len = [96, 192]
    # pred_len = [96, 192, 336]
    pred_len = [96, 192, 336, 720]
    # pred_len = [288, 432, 576]
    # save_dir = "/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings/"
    save_dir = root_pth
    all_df = []
    for d in ds:
        real_d = ds[d]
        ds_df = []
        # model_name = base_model_name + f"_bs{bs}"

        # hparam best
        # model_name = (
        #     base_model_name + "_bs256"
        #     if d in ["ETTm2", "weather"]
        #     else base_model_name + "_bs64"
        # )
        for pl in pred_len:
            # result_path = os.path.join(
            #     save_dir, f"{real_d}_{pl}_S", model_name, "dtm_.npy"
            # )

            # result_path = os.path.join(
            #     save_dir, f"{real_d}_{pl}_S", model_name, "initmodel_PatchTST_startks4__dtm__.npy"
            # )

            result_path = os.path.join(
                save_dir, f"{real_d}_{pl}_S", model_name, "cond_fcst_startks_None_fast_True_dtm_True.npy"
            )
            # result_path = os.path.join(
            #     save_dir, f"{real_d}_{pl}_S", model_name, "cond_fcst_startks_None_fast_True_dtm_True.npy"
            # )
            # result_path = os.path.join(
            #     save_dir, f"{real_d}_{pl}_S", model_name, "__dtm__.npy"
            # )
            results = np.load(result_path)
            df = pd.DataFrame(results, columns=["MAE", "MSE", "CRPS"])
            df = df.drop(columns=["MAE"])
            # print(df.std())
            df_mean = df.std()
            df_mean["method"] = model_name
            df_mean["pred_len"] = pl
            df_mean = pd.DataFrame(df_mean).T
            # df_mean = df_mean.set_index("method")
            ds_df.append(df_mean)
        ds_df = pd.concat(ds_df)

        # ds_df.index.name = "pred_len"
        ds_df["dataset"] = d
        # ds_df = ds_df.reset_index()
        ds_df = ds_df.set_index(["dataset", "pred_len"])
        all_df.append(ds_df)
    all_df = pd.concat(all_df)
    # print(all_df)
    # print(f"assets/{model_name}_std.csv")
    # all_df.to_csv(f"assets/{model_name}_fastfalse.csv")
    all_df.to_csv(f"assets/{model_name}_fcst_std.csv")

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
