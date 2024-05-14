from datetime import datetime
from functools import partial
from pathlib import Path

import argparse
from loguru import logger as log
from mmengine import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from logic.datamodules import EMSDataModule
from logic.io import read_raster_profile, write_raster

from logic.modules.single import SingleTaskModule
from logic.modules.multi import MultiModule
from logic.tiling.tilers import SmoothTiler, SimpleTiler
from logic.utils import exp_name_timestamp, find_best_checkpoint

import pathlib
import os

cli = argparse.ArgumentParser()
cli.add_argument("mode", choices=["train", "test", "test_full"])
cli.add_argument("-c", "--config_path", type=Path)
cli.add_argument("-e", "--experiment_path", type=Path)
cli.add_argument("-cp", "--checkpoint_path", type=Path)
cli.add_argument("-p", "--predict", action="store_true")
cli.add_argument("-ts", "--tile_size", type=int)
cli.add_argument("-tt", "--tiler_type", type=str, choices=["simple", "smooth"])
cli.add_argument("-sub", "--subdivisions", type=int)


def train(cfg_path: Path):
    log.info(f"Loading config from: {cfg_path}")
    config = Config.fromfile(cfg_path)
    # set the experiment name
    assert "name" in config, "Experiment name not specified in config."
    exp_name = exp_name_timestamp(config["name"])
    config["name"] = exp_name
    log.info(f"Experiment name: {exp_name}")

    if os.name == "nt":
        config.trainer.accelerator = "cpu"
        config.evaluation.accelerator = "cpu"
        config.data.root = "data\\little"

    in_channels = config["model"]["backbone"]["in_channels"]
    config["data"]["in_channels"] = in_channels

    reprojected = config["reprojected"] if "reprojected" in config else None
    layer_to_reproject = (
        config["layer_to_reproject"] if "layer_to_reproject" in config else None
    )
    assert reprojected and layer_to_reproject
    # datamodule
    log.info("Preparing the data module...")
    datamodule = EMSDataModule(**config["data"])

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    loss = config["loss"] if "loss" in config else "bce"
    module_class = (
        MultiModule
        if "aux_classes" in model_config["decode_head"]
        else SingleTaskModule
    )

    print(config)
    module = module_class(
        model_config,
        loss=loss,
        reprojected=reprojected,
        layer_to_reproject=layer_to_reproject,
    )
    module.init_pretrained()

    log.info("Preparing the trainer...")

    logger = TensorBoardLogger(save_dir="outputs", name=exp_name)
    config_dir = Path(logger.log_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config.dump(config_dir / "config.py")
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights",
            monitor="val_loss",
            mode="min",
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=6,
            every_n_epochs=10,
        )
    ]
    trainer = Trainer(
        **config["trainer"], callbacks=callbacks, logger=logger, num_sanity_val_steps=0
    )

    log.info("Starting the training...")
    trainer.fit(module, datamodule=datamodule)


def test(exp_path: Path, checkpoint: Path, predict: bool):

    log.info(f"Loading experiment from: {exp_path}")

    config_path = exp_path / "config.py"
    models_path = exp_path / "weights"
    # asserts to check the experiment folders
    assert exp_path.exists(), "Experiment folder does not exist."
    assert config_path.exists(), f"Config file not found in: {config_path}"
    assert models_path.exists(), f"Models folder not found in: {models_path}"
    # load training config
    config = Config.fromfile(config_path)
    if os.name == "nt":
        config.trainer.accelerator = "cpu"
        config.evaluation.accelerator = "cpu"
    # datamodule
    log.info("Preparing the data module...")
    in_channels = config["model"]["backbone"]["in_channels"]
    config["data"]["in_channels"] = in_channels
    datamodule = EMSDataModule(**config["data"])

    # prepare the model
    checkpoint = checkpoint or find_best_checkpoint(models_path, "val_loss", "min")
    log.info(f"Using checkpoint: {checkpoint}")
    log.info(str(type(checkpoint)))

    module_opts = dict(config=config["model"])
    loss = config["loss"] if "loss" in config else "bce"
    module_opts.update(loss=loss)
    if predict:
        tiler = SimpleTiler(
            tile_size=config["data"]["patch_size"],
            batch_size=config["data"]["batch_size_eval"],
            channels_first=True,
            mirrored=False,
        )
        output_path = exp_path / "predictions"
        output_path.mkdir(parents=True, exist_ok=True)
        inference_fn = partial(process_inference, output_path=output_path)
        module_opts.update(tiler=tiler, predict_callback=inference_fn)

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    module_class = (
        MultiModule
        if "aux_classes" in model_config["decode_head"]
        else SingleTaskModule
    )
    log.info(f"versione stringa:{type(str(checkpoint))}")

    if os.name == "nt":
        backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            module = module_class.load_from_checkpoint(
                str(checkpoint), "cpu", **module_opts
            )
        finally:
            pathlib.PosixPath = backup
    else:
        module = module_class.load_from_checkpoint(str(checkpoint), **module_opts)

    logger = TensorBoardLogger(
        save_dir="outputs", name=config["name"], version=exp_path.stem
    )
    if predict:
        log.info("Generating predictions...")
        trainer = Trainer(**config["evaluation"], logger=False)
        trainer.predict(module, datamodule=datamodule, return_predictions=False)
    else:
        log.info("Starting the testing...")
        trainer = Trainer(**config["evaluation"], logger=logger)
        trainer.test(module, datamodule=datamodule)


def test_full(
    exp_path: Path,
    checkpoint: Path,
    tile_size: int,
    subdivisions,
    tiler_type="simple",
):
    log.info(f"Test full images")
    log.info(f"Loading experiment from: {exp_path}")

    config_path = exp_path / "config.py"
    models_path = exp_path / "weights"

    # asserts to check the experiment folders
    assert exp_path.exists(), "Experiment folder does not exist."
    assert config_path.exists(), f"Config file not found in: {config_path}"
    assert models_path.exists(), f"Models folder not found in: {models_path}"

    # load training config
    config = Config.fromfile(config_path)
    if os.name == "nt":
        config.trainer.accelerator = "cpu"
        config.evaluation.accelerator = "cpu"
    # default arguments
    tile_size = tile_size if tile_size else config["data"]["patch_size"]
    subdivisions = subdivisions if subdivisions else 2
    tiler_type = tiler_type if tiler_type else "simple"
    assert tiler_type in ["simple", "smooth"]

    # datamodule
    log.info("Preparing the data module...")
    in_channels = config["model"]["backbone"]["in_channels"]
    config["data"]["in_channels"] = in_channels
    config["data"]["full_test_type"] = True
    datamodule = EMSDataModule(**config["data"])

    # prepare the model
    checkpoint = checkpoint or find_best_checkpoint(models_path, "val_loss", "min")
    log.info(f"Using checkpoint: {checkpoint}")

    module_opts = dict(config=config["model"])
    loss = config["loss"] if "loss" in config else "bce"
    module_opts.update(loss=loss)
    log.info(
        f"Tile size: {tile_size}, subdivisions: {subdivisions}, tiler type: {tiler_type}"
    )
    TilerClass = SmoothTiler if tiler_type == "smooth" else SimpleTiler
    tiler = TilerClass(
        tile_size=tile_size,
        batch_size=config["data"]["batch_size_eval"],
        channels_first=True,
        mirrored=False,
        subdivisions=subdivisions,
    )
    module_opts.update(tiler=tiler)
    module_opts.update(test_type="full")

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    module_class = (
        MultiModule
        if "aux_classes" in model_config["decode_head"]
        else SingleTaskModule
    )

    if os.name == "nt":
        backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            module = module_class.load_from_checkpoint(
                str(checkpoint), "cpu", **module_opts
            )
        finally:
            pathlib.PosixPath = backup
    else:
        module = module_class.load_from_checkpoint(str(checkpoint), **module_opts)

    logger = TensorBoardLogger(
        save_dir="outputs", name=config["name"], version=exp_path.stem
    )

    log.info("Starting the testing...")
    trainer = Trainer(**config["evaluation"], logger=logger)
    trainer.predict(module, datamodule=datamodule)


def process_inference(
    batch: dict,
    output_path: Path,
):
    assert output_path.exists(), f"Output path does not exist: {output_path}"
    # for binary segmentation
    prediction = (batch["pred"] > 0.5).int().unsqueeze(0)
    prediction = prediction.cpu().numpy()
    # store the prediction as a GeoTIFF, reading the spatial information from the input image
    image_path = Path(batch["metadata"]["S2L2A"][0])
    input_profile = read_raster_profile(image_path)
    output_profile = input_profile.copy()
    output_profile.update(dtype="uint8", count=1)
    output_file = output_path / f"{image_path.stem}.tif"
    write_raster(path=output_file, data=prediction, profile=output_profile)


if __name__ == "__main__":
    seed_everything(95, workers=True)

    if os.name == "nt":
        # test_str = "test_full -e outputs\\upernet-rn50_single_ssl4eo_50ep_20240313_153424\\version_0 -ts 128 -tt smooth -sub 8"
        # args = cli.parse_args(test_str.split())
        train_swin = "train -c configs\\single\\pretrained\\dice\\ems_upernet-rn50_single_10ep_6ch.py"
        args = cli.parse_args(train_swin.split())
        print(args)
    else:
        args = cli.parse_args()

    match args.mode:
        case "train":
            train(cfg_path=args.config_path)
        case "test":
            test(
                exp_path=args.experiment_path,
                checkpoint=args.checkpoint_path,
                predict=args.predict,
            )
        case "test_full":
            test_full(
                exp_path=args.experiment_path,
                checkpoint=args.checkpoint_path,
                tile_size=args.tile_size,
                subdivisions=args.subdivisions,
                tiler_type=args.tiler_type,
            )

    """cli(
        "train -c configs/single/pretrained/ems_upernet-rn50_single_10ep_16ch.py".split()
    )"""
