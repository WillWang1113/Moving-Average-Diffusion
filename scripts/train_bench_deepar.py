import logging
import os
import argparse
import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import DeepAREstimator

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    df = pd.read_csv(
        "/home/user/data/FrequencyDiffusion/dataset/MFRED_clean.csv",
        index_col=0,
        parse_dates=True,
    )[: 288*10]
    df = df.drop(columns=["weekday", "hour"])
    print(df)
    prediction_length = args['horizon']
    train_data = PandasDataset(
    df[:-prediction_length*2],target='value',
    feat_dynamic_real=["t2m"],
    )

    test_data = PandasDataset(
        df,target='value',
        feat_dynamic_real=["t2m"],
    )
        
    estimator_with_features = DeepAREstimator(
        freq="5min",
        prediction_length=args["horizon"],
        num_feat_dynamic_real=1,
        trainer_kwargs={"max_epochs": 1},
    )
    predictor = estimator_with_features.train(train_data)
    # forecast_it, ts_it = make_evaluation_predictions(dataset=test, predictor=predictor)
    
    # _, test_template = split(test, offset=-args['horizon'])
    # test_data = test_template.generate_instances(args['horizon'])
    # print('begin predicting!')
    
    # forecast_it = predictor.predict(test_data.input)
    # print('finish predicting!')
    # fig, ax = plt.subplots()
    # for forecast in forecast_it:
    #     forecast.plot(ax=ax)
    #     break
    # ax.legend(["True values"], loc="upper left", fontsize="xx-large")
    # fig.savefig('test.png')
    
    # return 0

    # # Train the model and make predictions
    # model = DeepAREstimator(
    #     use_feat_dynamic_real=True,
    #     context_length=args["horizon"],
    #     prediction_length=args["horizon"],
    #     freq="5min",
    #     trainer_kwargs={"max_epochs": 5},
    # ).train(training_data)

    # out_dict = {}
    # for i, m in enumerate(model_names):
    #     print(m)
    #     model_cols = [c for c in cols if m in c]
    #     print(model_cols)
    #     print(len(model_cols))
    #     # print(Y_hat_df[model_cols])

    #     y_hat = Y_hat_df[model_cols].values
    #     y_hat = y_hat.reshape(-1, horizon, n_series, len(quantiles))
    #     # print(y_hat.shape)
    #     # print(y_true.shape)
    #     # y_hat = y_hat.sort(axis=-1)

    #     (RMSE, MAE, PBL) = get_bench_metrics(
    #         y_hat, y_true, quantiles=np.array(quantiles)
    #     )
    #     out_dict[m] = (RMSE, MAE, PBL)
    #     # fig, ax = plt.subplots()
    #     # ax.plot(y_true[127])
    #     # ax.plot(y_hat[127].squeeze())
    #     # fig.savefig("test.png")


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
    print(all_out_dict)
    # with open(
    #     "/home/user/data/FrequencyDiffusion/savings/mfred/benchmarks/deepar_metrics.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(all_out_dict, f)
