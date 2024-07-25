import argparse
import os
from lightning.fabric import seed_everything
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import OffsetSplitter
from gluonts.evaluation import (
    make_evaluation_predictions,
    Evaluator,
    MultivariateEvaluator,
)
from gluonts.torch.model.d_linear import DLinearEstimator
from gluonts.torch.model.i_transformer import ITransformerEstimator
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from src.utils.train import (
    ConcatDataset,
    filter_metrics,
)
import pandas as pd


def main(args, n):
    seed_everything(n, workers=True)

    dataset = get_dataset(args["dataset"])
    num_rolling_evals = int(len(dataset.test) / len(dataset.train))

    # train_grouper = MultivariateGrouper(
    #     max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
    # )

    # test_grouper = MultivariateGrouper(
    #     num_test_dates=num_rolling_evals,
    #     max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    # )
    context_length = args["seq_len"]
    prediction_length = args["pred_len"]

    train_val_splitter = OffsetSplitter(offset=-prediction_length * num_rolling_evals)
    _, val_gen = train_val_splitter.split(dataset.train)

    val_dataset = ConcatDataset(
        val_gen.generate_instances(prediction_length, num_rolling_evals)
    )

    # train_data = train_grouper(dataset.train)
    train_data = dataset.train
    val_data = val_dataset
    # test_data = test_grouper(dataset.test)
    test_data = dataset.test
    trainer_kwargs = dict(
        max_epochs=args["epochs"],
        deterministic=True,
        devices=[args["gpu"]],
        default_root_dir=os.path.join(args["save_dir"], args["dataset"]),
    )
    models = [
        (
            "DLinear",
            DLinearEstimator(
                prediction_length=prediction_length,
                context_length=context_length,
                batch_size=args["batch_size"],
                num_batches_per_epoch=args["num_batches_per_epoch"],
                trainer_kwargs=trainer_kwargs,
                scaling="std",
            ),
        ),
        (
            "PatchTST",
            PatchTSTEstimator(
                prediction_length=prediction_length,
                context_length=context_length,
                patch_len=16,
                batch_size=args["batch_size"],
                num_batches_per_epoch=args["num_batches_per_epoch"],
                trainer_kwargs=trainer_kwargs,
                scaling="std",
            ),
        ),
        # (
        #     "ITransformer",
        #     ITransformerEstimator(
        #         prediction_length=prediction_length,
        #         context_length=context_length,
        #         batch_size=args["batch_size"],
        #         num_batches_per_epoch=args["num_batches_per_epoch"],
        #         trainer_kwargs=trainer_kwargs,
        #         scaling="std",
        #     ),
        # ),
    ]
    all_metrics = []
    for (name, m) in models:
        m_predictor = m.train(train_data, val_data, cache_data=True)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data, predictor=m_predictor, num_samples=args["num_samples"]
        )

        forecasts = list(f.to_sample_forecast() for f in forecast_it)
        # forecasts = list(forecast_it)
        tss = list(ts_it)
        evaluator = Evaluator()
        # evaluator = MultivariateEvaluator()
        metrics, _ = evaluator(tss, forecasts)
        metrics = [metrics["mean_wQuantileLoss"],metrics["NRMSE"]]
        all_metrics.append({name: metrics})
    all_metrics = pd.DataFrame(all_metrics, aixs=1)
    print(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_feat_dynamic_real", type=int, default=0)
    parser.add_argument("--num_feat_static_real", type=int, default=0)
    parser.add_argument("--num_feat_static_cat", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=100)
    # parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    for i in range(args.num_train):
        main(vars(args), i)
        if args.smoke_test:
            break
