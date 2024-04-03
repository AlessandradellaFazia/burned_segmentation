from datetime import datetime
from functools import partial
from pathlib import Path

from argdantic import ArgField, ArgParser
from loguru import logger as log
from mmengine import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from logic.datamodules import EMSDataModule
from logic.io import read_raster_profile, write_raster

from logic.modules.single import SingleTaskModule
from logic.tiling.tilers import SmoothTiler
from logic.utils import exp_name_timestamp, find_best_checkpoint

import pathlib


cli = ArgParser()


@cli.command()
def train(
    cfg_path: Path = ArgField("-c", description="Path to the config file."),
    keep_name: bool = ArgField(
        "-k",
        default=False,
        description="Keep the experiment name as specified in the config file.",
    ),
):
    log.info(f"Loading config from: {cfg_path}")
    config = Config.fromfile(cfg_path)
    # set the experiment name
    assert "name" in config, "Experiment name not specified in config."
    exp_name = exp_name_timestamp(config["name"]) if not keep_name else config["name"]
    config["name"] = exp_name
    log.info(f"Experiment name: {exp_name}")

    # datamodule
    log.info("Preparing the data module...")
    datamodule = EMSDataModule(**config["data"])

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    loss = config["loss"] if "loss" in config else "bce"
    reprojected = config["reprojected"] if "reprojected" in config else None

    module = SingleTaskModule(model_config, loss=loss, reprojected=reprojected)
    module.init_pretrained()

    log.info("Preparing the trainer...")

    logger = TensorBoardLogger(save_dir="outputs", name=exp_name)
    config_dir = Path(logger.log_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config.dump(config_dir / "config.py")
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights",
            monitor="epoch",
            mode="max",
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


@cli.command()
def test(
    exp_path: Path = ArgField("-e", description="Path to the experiment folder."),
    checkpoint: Path = ArgField(
        "-c",
        default=None,
        description="Path to the checkpoint file. If not specified, the best checkpoint will be loaded.",
    ),
    predict: bool = ArgField(
        default=False, description="Generate predictions on the test set."
    ),
):

    log.info(f"Loading experiment from: {exp_path}")
    log.info(str(type(exp_path)))

    config_path = exp_path / "config.py"
    models_path = exp_path / "weights"
    # asserts to check the experiment folders
    assert exp_path.exists(), "Experiment folder does not exist."
    assert config_path.exists(), f"Config file not found in: {config_path}"
    assert models_path.exists(), f"Models folder not found in: {models_path}"
    # load training config
    config = Config.fromfile(config_path)

    # datamodule
    log.info("Preparing the data module...")
    datamodule = EMSDataModule(**config["data"])

    # prepare the model
    checkpoint = checkpoint or find_best_checkpoint(models_path, "val_loss", "min")
    log.info(f"Using checkpoint: {checkpoint}")
    log.info(str(type(checkpoint)))

    module_opts = dict(config=config["model"])
    loss = config["loss"] if "loss" in config else "bce"
    module_opts.update(loss=loss)
    if predict:
        tiler = SmoothTiler(
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

    log.info(f"versione stringa:{type(str(checkpoint))}")
    backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        module = SingleTaskModule.load_from_checkpoint(
            str(checkpoint), "cpu", **module_opts
        )
    finally:
        pathlib.PosixPath = backup
    print(module)

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
    # cli()
    # cli("train -c configs\single\pretrained\swin\swin.py".split())
    cli("train -c configs\single\pretrained\ems_upernet-rn50_single_50ep.py".split())
    """cli(
        "test -e outputs\\upernet-rn50_single_no_pre_50ep_20240304_153937\\version_0 -c outputs\\upernet-rn50_single_no_pre_50ep_20240304_153937\\version_0\\weights\\model-epoch=02-val_loss=0.08.ckpt".split()
    )"""

    """cli(
        "test -e outputs\\upernet-rn50_single_no_pre_50ep_20240304_153937\\version_0 -c outputs\\upernet-rn50_single_no_pre_50ep_20240304_153937\\version_0\\weights\\model-epoch=02-val_loss=0.08.ckpt --predict".split()
    )"""
