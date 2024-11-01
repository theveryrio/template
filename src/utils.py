# Author: Chanho Kim <theveryrio@gmail.com>

from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")

    hparams["task_name"] = cfg.get("task_name")
    hparams["seed"] = cfg.get("seed")

    data_info = [
        "train_csv_file",
        "val_csv_file",
        "test_csv_file",
        "predict_csv_file",
        "x_columns",
        "y_columns",
    ]

    for key in data_info:
        hparams[key] = cfg[key]

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def create(config_name: str) -> Tuple[LightningModule, LightningDataModule, Trainer]:
    """A simplified version of the "create_lightning" function.

    :param config_name: The name of the config (usually the file name without the .yaml extension).
    :return: The tuple with LightningModule, LightningDataModule, and Trainer.
    """
    obj_dict = create_lightning(config_name)
    return obj_dict["model"], obj_dict["datamodule"], obj_dict["trainer"]


def create_lightning(config_name: str) -> Dict[str, Any]:
    """
    :param config_name: The name of the config
           (usually the file name without the .yaml extension).
    :return: The dict with all instantiated objects.
    """
    with hydra.initialize(version_base="1.3", config_path="."):
        default_cfg = hydra.compose(config_name="default.yaml")

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)

    # Disable strict mode and merge the two configurations
    with open_dict(default_cfg):
        cfg = OmegaConf.merge(default_cfg, cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log_hyperparameters(object_dict)

    return object_dict
