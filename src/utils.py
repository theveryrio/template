# Author: Chanho Kim <theveryrio@gmail.com>

import functools
import re
from typing import Any, Dict, List, Tuple, Union

import hydra
import lightning as L
import optuna
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

optuna.logging.set_verbosity(optuna.logging.ERROR)


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


def suggest_hyperparameter(trial: optuna.Trial, name: str, param: list) -> Union[int, float]:
    """Suggests a hyperparameter value based on the specified type and parameters.
    :param trial: An Optuna trial object used to suggest values for hyperparameters.
    :param name: The name of the hyperparameter to suggest.
    :param param: A list where the first element specifies the type of suggestion 
        ('choice', 'log', or 'uniform'), followed by the necessary parameters for that suggestion type.
    :raises ValueError: If `param` has fewer than 3 elements.
    :raises ValueError: If the first element of `param` is not one of 'choice', 'log', or 'uniform'.
    :return: int or float: The suggested hyperparameter value.
    """
    # Validate the length of param
    if len(param) < 3:
        raise ValueError("Parameter must contain at least 3 elements.")
    valid_param_types = ["choice", "log", "uniform"]
 
    # Validate the parameter type
    param_type = param[0]
    if param_type not in valid_param_types:
        raise ValueError(f"Invalid param_type. Choose from {valid_param_types}.")
 
    # Determine the suggestion method and casting function
    if param[1].isdecimal():
        suggest_method = trial.suggest_int
        cast_func = int
    else:
        suggest_method = trial.suggest_float
        cast_func = float
 
    # Suggest hyperparameters based on the type
    if param_type == "choice":
        return trial.suggest_categorical(name, [cast_func(p) for p in param[1:]])
    elif param_type == "log":
        return suggest_method(name, cast_func(param[1]), cast_func(param[2]), log=True)
    elif param_type == "uniform":
        if len(param) > 3:
            return suggest_method(name, cast_func(param[1]), cast_func(param[2]), step=cast_func(param[3]))
        return suggest_method(name, cast_func(param[1]), cast_func(param[2]))


def apply_optuna_suggestions(trial: optuna.Trial, cfg: DictConfig) -> None:
    """Apply hyperparameter suggestions from an Optuna trial to a given configuration.
 
    :param trial: An Optuna trial object used to suggest values for hyperparameters.
    :param cfg: A hierarchical configuration object from OmegaConf.
    """
    for key, value in cfg.items():
        if isinstance(value, dict) or OmegaConf.is_dict(value):
            apply_optuna_suggestions(trial, value)
        elif isinstance(value, str):
            param = [p for p in re.split(r'[(),\s]+', value) if p]
            if param[0] in ["choice", "log", "uniform"]:
                cfg[key] = suggest_hyperparameter(trial, key, param)


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """Objective function for Optuna to minimize.
 
    :param trial: An Optuna trial object used to suggest values for hyperparameters.
    :param cfg: A hierarchical configuration object from OmegaConf.
    :return: The computed objective value.
    """
    cfg_copy = OmegaConf.merge(cfg)
    apply_optuna_suggestions(trial, cfg_copy)
    obj_dict = create_lightning(cfg_copy)
    model = obj_dict["model"]
    datamodule = obj_dict["datamodule"]
    trainer = obj_dict["trainer"]
    trainer.fit(model, datamodule)
    return trainer.test(model, datamodule)[0][cfg.objective]


def train_lightning(cfg: DictConfig) -> None:
    """
    :param cfg: A hierarchical configuration object from OmegaConf.
    """
    obj_dict = create_lightning(cfg)
    model = obj_dict["model"]
    datamodule = obj_dict["datamodule"]
    trainer = obj_dict["trainer"]
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


def train_lightning_with_optuna(cfg: DictConfig) -> None:
    """
    :param cfg: A hierarchical configuration object from OmegaConf.
    """
    study = hydra.utils.instantiate(cfg.optuna)
    study.optimize(functools.partial(objective, cfg=cfg), n_trials=cfg.n_trials)


def create_hydra_cfg(config_name: str) -> DictConfig:
    """
    :param config_name: The name of the config (usually the file name without the .yaml extension).
    :return: The dict with all instantiated objects.
    """
    with hydra.initialize(version_base="1.3", config_path="."):
        default_cfg = hydra.compose(config_name="default.yaml")
 
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)
 
    # Disable strict mode and merge the two configurations
    with open_dict(default_cfg):
        return OmegaConf.merge(default_cfg, cfg)


def create_lightning(cfg: DictConfig) -> Dict[str, Any]:
    """
    :param cfg: A hierarchical configuration object from OmegaConf.
    :return: The dict with all instantiated objects.
    """
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


def train(config_name: str) -> None:
    """
    :param config_name: The name of the config (usually the file name without the .yaml extension).
    """
    cfg = create_hydra_cfg(config_name)
    if cfg.n_trials:
        train_lightning_with_optuna(cfg)
    else:
        train_lightning(cfg)


def create(config_name: str) -> Tuple[LightningModule, LightningDataModule, Trainer]:
    """
    :param config_name: The name of the config (usually the file name without the .yaml extension).
    :return: The tuple with LightningModule, LightningDataModule, and Trainer.
    """
    cfg = create_hydra_cfg(config_name)
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
 
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)
 
    return model, datamodule, trainer
