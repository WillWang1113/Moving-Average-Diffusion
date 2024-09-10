import argparse
import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import yaml
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from gluonts.dataset.repository import get_dataset
from src.utils.train import (
    ConcatDataset,
    filter_metrics,
    create_splitter,
    create_transforms,
)

from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator


from src.utils.parser import exp_parser
from src.models.diffusion_gts import MAD
from src.utils.schedule import get_schedule


def prepare_train(model_config, args, n, dataloader):
    root_pth = args["save_dir"]
    df_ = model_config["diff_config"].pop("name")
    # df = getattr(diffusion_gts, df_)
    df = MAD
    print(df)
    data_folder = os.path.join(
        root_pth,
        f"{args['dataset']}_{args['seq_len']}_{args['pred_len']}",
    )
    save_folder = os.path.join(data_folder, args["model_config"])
    # save_folder = os.path.join(data_folder, df_, exp_name)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(model_config, w, indent=2)

    # with open(os.path.join(data_folder, "scaler.npy"), "wb") as f:
    #     np.save(f, scaler)
    # torch.save(test_dl, os.path.join(data_folder, "test_dl.pt"))
    batch = next(iter(dataloader))
    seq_length, seq_channels = (
        batch["past_target"].shape[1],
        1,
        # batch["past_target"].shape[2],
    )
    target_seq_length, target_seq_channels = (
        batch["future_target"].shape[1],
        1,
        # batch["future_target"].shape[2],
    )
    ff = batch.get("future_features", None)
    future_seq_length, future_seq_channels = (
        ff.shape[1] if ff is not None else ff,
        ff.shape[2] if ff is not None else ff,
    )

    model_config["bb_config"]["seq_channels"] = seq_channels
    model_config["bb_config"]["seq_length"] = target_seq_length
    model_config["bb_config"]["latent_dim"] = model_config["cn_config"]["latent_dim"]
    model_config["cn_config"]["seq_channels"] = seq_channels
    model_config["cn_config"]["seq_length"] = seq_length
    model_config["cn_config"]["future_seq_channels"] = future_seq_channels
    model_config["cn_config"]["future_seq_length"] = future_seq_length
    model_config["cn_config"]["target_seq_length"] = target_seq_length
    model_config["cn_config"]["target_seq_channels"] = target_seq_channels
    ns_name = model_config["diff_config"].pop("noise_schedule")
    n_steps = (
        target_seq_length - 1
        if df_.__contains__("MAD")
        else model_config["diff_config"]["T"]
    )
    noise_schedule = get_schedule(ns_name, n_steps)
    return model_config, noise_schedule, df, save_folder


def main(args, n):
    seed_everything(n, workers=True)
    # args = yaml.safe_load(
    #     open(f'configs/dataset/{args["args"]}.yaml', "r")
    # )
    # args = exp_parser(args, args)
    # # data_fn = getattr(dataset, args.pop("data_fn"))

    model_config = yaml.safe_load(
        open(f'configs/model/{args["model_config"]}.yaml', "r")
    )
    model_config = exp_parser(model_config, args)

    dataset = get_dataset(args["dataset"])
    num_rolling_evals = int(len(dataset.test) / len(dataset.train))

    context_length = args["seq_len"]
    prediction_length = args["pred_len"]

    transformation = create_transforms(
        num_feat_dynamic_real=args["num_feat_dynamic_real"],
        num_feat_static_real=args["num_feat_static_real"],
        num_feat_static_cat=args["num_feat_static_cat"],
    )

    training_splitter = create_splitter(context_length, prediction_length, "train")
    train_dl = TrainDataLoader(
        # We cache the dataset, to make training faster
        Cached(dataset.train),
        batch_size=args["batch_size"],
        stack_fn=batchify,
        transform=transformation + training_splitter,
        num_batches_per_epoch=args['num_batches_per_epoch'],
    )
    

    train_val_splitter = OffsetSplitter(offset=-prediction_length * num_rolling_evals)
    _, val_gen = train_val_splitter.split(dataset.train)

    val_dataset = ConcatDataset(
        val_gen.generate_instances(prediction_length, num_rolling_evals)
    )
    val_splitter = create_splitter(
        past_length=context_length,
        future_length=prediction_length,
        mode="val",
    )
    # transformed_valdata = transformation.apply(val_dataset, is_train=True)
    val_dl = ValidationDataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        stack_fn=batchify,
        transform=transformation + val_splitter,
    )

    prediction_splitter = create_splitter(
        past_length=context_length,
        future_length=prediction_length,
        mode="test",
    )

    model_config, noise_schedule, df, save_folder = prepare_train(
        model_config, args, n, train_dl
    )

    diff = df(
        backbone_config=model_config["bb_config"],
        conditioner_config=model_config["cn_config"],
        noise_schedule=noise_schedule,
        lr=model_config["train_config"]["lr"],
        alpha=model_config["train_config"]["alpha"],
        **model_config["diff_config"],
    )

    # es = EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     patience=model_config["train_config"]["early_stop"],
    # )
    monitor = 'train_loss'
    mc = ModelCheckpoint(monitor=monitor, mode="min", verbose=True)
    # # return 0
    # # TODO:
    trainer = Trainer(
        max_epochs=args["epochs"],
        deterministic=True,
        devices=[args["gpu"]],
        callbacks=[mc],
        gradient_clip_val=args["grad_clip_value"],
        # callbacks=[es, mc],
        default_root_dir=save_folder,
        # **model_config["train_config"],
    )

    trainer.fit(diff, train_dl)
    # trainer.fit(diff, train_dl, val_dl)
    # torch.save(mc.best_model_path, os.path.join(save_folder, f'best_model_path_{n}.pt'))
    # diff = df.load_from_checkpoint(checkpoint_path='/home/user/data/FrequencyDiffusion/savings/electricity_nips_96_96/MADfreq_gts/lightning_logs/version_16/checkpoints/epoch=66-step=6700.ckpt')
    diff = df.load_from_checkpoint(checkpoint_path=mc.best_model_path)
    diff.config_sampling(n_sample=args['num_samples'])
    diff_predictor = diff.get_predictor(transformation + prediction_splitter)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, predictor=diff_predictor, num_samples=args['num_samples']
    )

    # forecasts_pytorch = list(f.to_sample_forecast() for f in forecast_it)
    # tss_pytorch = list(ts_it)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    fig, axs = plt.subplots(2,2)
    axs = axs.flatten()
    for i in range(len(axs)):
        axs[i].plot(tss[i][-args['seq_len']-args['pred_len']:].to_timestamp())
        forecasts[i].plot(show_label=True, ax=axs[i])
        axs[i].legend()
    fig.tight_layout()
    fig.savefig('results.png')
    evaluator = Evaluator()
    metrics, _ = evaluator(tss, forecasts)
    metrics = [metrics["mean_wQuantileLoss"], metrics["NRMSE"]]
    all_metrics = {"MAD-F": metrics}
    all_metrics = pd.DataFrame(all_metrics, index=["mean_wQuantileLoss", "NRMSE"])
    all_metrics.to_csv(
        os.path.join(
            args["save_dir"],
            args["dataset"] + f"_{args['seq_len']}_{args['pred_len']}",
            f"metric_{n}.csv",
        )
    )
    # # pred = torch.concat(pred)
    # # print(pred.shape)

    # print(mc.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters config")
    parser.add_argument(
        "-mc",
        "--model_config",
        required=True,
        help="name of model configuration file.",
    )
    # parser.add_argument(
    #     "-dc",
    #     "--args",
    #     type=str,
    #     required=True,
    #     help="name of data configuration file.",
    # )
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)

    # Define overrides on models
    parser.add_argument("--diff_config.name", type=str)
    parser.add_argument("--diff_config.norm", action="store_true", default=None)
    parser.add_argument("--diff_config.pred_diff", action="store_true", default=None)
    parser.add_argument("--diff_config.noise_schedule", type=str)
    parser.add_argument("--bb_config.name", type=str)
    parser.add_argument("--bb_config.hidden_size", type=int)

    # Define overrides on dataset
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_feat_dynamic_real", type=int, default=0)
    parser.add_argument("--num_feat_static_real", type=int, default=0)
    parser.add_argument("--num_feat_static_cat", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--grad_clip_value", type=float, default=1.0)
    args = parser.parse_args()
    for i in range(args.num_train):
        main(vars(args), i)
        if args.smoke_test:
            break
