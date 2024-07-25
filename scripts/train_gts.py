import argparse
import json
import os
import yaml
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from gluonts.dataset.repository import get_dataset
from src.utils.train import ConcatDataset, filter_metrics, create_splitter, create_transforms

from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator


from src.utils.parser import exp_parser
from src.models import diffusion_gts
from src.utils.schedule import get_schedule


def prepare_train(model_config, data_config, args, n, dataloader):
    root_pth = args["save_dir"]
    df_ = model_config["diff_config"].pop("name")
    df = getattr(diffusion_gts, df_)
    print(df)
    data_folder = os.path.join(
        root_pth,
        f"{args['data_config']}_{data_config['pred_len']}_{data_config['features']}",
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
        batch["past_target"].shape[2],
    )
    target_seq_length, target_seq_channels = (
        batch["future_target"].shape[1],
        batch["future_target"].shape[2],
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
    noise_schedule = get_schedule(ns_name, args["data_config"], n_steps)
    return model_config, noise_schedule, df, save_folder


def main(args, n):
    seed_everything(n, workers=True)
    data_config = yaml.safe_load(
        open(f'configs/dataset/{args["data_config"]}.yaml', "r")
    )
    data_config = exp_parser(data_config, args)
    # data_fn = getattr(dataset, data_config.pop("data_fn"))

    model_config = yaml.safe_load(
        open(f'configs/model/{args["model_config"]}.yaml', "r")
    )
    model_config = exp_parser(model_config, args)

    dataset = get_dataset(args["dataset"])
    num_rolling_evals = int(len(dataset.test) / len(dataset.train))

    context_length = data_config["seq_len"]
    prediction_length = data_config["pred_len"]

    transformation = create_transforms(
        num_feat_dynamic_real=args["num_feat_dynamic_real"],
        num_feat_static_real=args["num_feat_static_real"],
        num_feat_static_cat=args["num_feat_static_cat"],
    )

    training_splitter = create_splitter(context_length, prediction_length, "train")
    train_dl = TrainDataLoader(
        # We cache the dataset, to make training faster
        Cached(dataset.train),
        batch_size=data_config["batch_size"],
        stack_fn=batchify,
        transform=transformation + training_splitter,
        num_batches_per_epoch=1,
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
        batch_size=data_config["batch_size"] * 10,
        stack_fn=batchify,
        transform=transformation + val_splitter,
    )

    prediction_splitter = create_splitter(
        past_length=context_length,
        future_length=prediction_length,
        mode="test",
    )

    model_config, noise_schedule, df, save_folder = prepare_train(
        model_config, data_config, args, n, train_dl
    )
    # # ! MUST SETUP SEED AFTER prepare_train
    # # setup_seed(n)
    diff = df(
        backbone_config=model_config["bb_config"],
        conditioner_config=model_config["cn_config"],
        noise_schedule=noise_schedule,
        lr=model_config["train_config"]["lr"],
        alpha=model_config["train_config"]["alpha"],
        **model_config["diff_config"],
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=model_config["train_config"]["early_stop"],
    )
    mc = ModelCheckpoint(monitor="val_loss", dirpath=save_folder, save_top_k=1)
    # return 0
    # TODO:
    trainer = Trainer(
        max_epochs=model_config["train_config"]["epochs"],
        deterministic=True,
        devices=[args["gpu"]],
        # callbacks=[mc],
        callbacks=[es, mc],
        default_root_dir=save_folder,
        # **model_config["train_config"],
    )

    trainer.fit(diff, train_dl, val_dl)

    # torch.save(mc.best_model_path, os.path.join(save_folder, f'best_model_path_{n}.pt'))
    diff = df.load_from_checkpoint(checkpoint_path=mc.best_model_path)
    diff.config_sampling()
    diff_predictor = diff.get_predictor(transformation + prediction_splitter)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, predictor=diff_predictor
    )

    # forecasts_pytorch = list(f.to_sample_forecast() for f in forecast_it)
    # tss_pytorch = list(ts_it)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator()
    metrics, _ = evaluator(tss, forecasts)
    metrics = filter_metrics(metrics)
    print(metrics)
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
    parser.add_argument(
        "-dc",
        "--data_config",
        type=str,
        required=True,
        help="name of data configuration file.",
    )
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
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--num_feat_dynamic_real", type=int, default=0)
    parser.add_argument("--num_feat_static_real", type=int, default=0)
    parser.add_argument("--num_feat_static_cat", type=int, default=0)
    args = parser.parse_args()
    for i in range(args.num_train):
        main(vars(args), i)
        if args.smoke_test:
            break
