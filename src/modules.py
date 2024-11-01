# Author: Chanho Kim <theveryrio@gmail.com>

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule


class Module(LightningModule):
    """A module for automatic optimization."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """
        :param model: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.model`.

        :param x: An input tensor representing the data to be passed through the model.
        :return: An output tensor produced by the model, containing the model's activations or
            predictions.
        """
        return self.model(x)

    def move_metrics_to_device(self):
        """Moves all metrics in the `self.model.metrics` dictionary to the specified device.

        This ensures that metric computations occur on the correct device (e.g., GPU/CPU).
        """
        for key, metric in self.model.metrics.items():
            self.model.metrics[key] = metric.to(self.device)

    def reset_metrics(self):
        """Resets all metrics in the `self.model.metrics` dictionary.

        This is useful for clearing any accumulated state before a new epoch starts.
        """
        for key, metric in self.model.metrics.items():
            metric.reset()

    def on_train_start(self):
        """Lightning hook that is called when training begins.

        Ensures that all metrics are moved to the correct device before training begins.
        """
        self.move_metrics_to_device()

    def on_validation_start(self):
        """Lightning hook that is called when validation begins.

        Ensures that all metrics are moved to the correct device before validation begins.
        """
        self.move_metrics_to_device()

    def on_test_start(self):
        """Lightning hook that is called when test begins.

        Ensures that all metrics are moved to the correct device before test begins.
        """
        self.move_metrics_to_device()

    def on_train_epoch_start(self):
        """Lightning hook that is called when a training epoch starts.

        Resets the state of all metrics to ensure they only reflect the current epoch.
        """
        self.reset_metrics()

    def on_validation_epoch_start(self):
        """Lightning hook that is called when a validation epoch starts.

        Resets the state of all metrics to ensure they only reflect the current epoch.
        """
        self.reset_metrics()

    def on_test_epoch_start(self):
        """Lightning hook that is called when a test epoch starts.

        Resets the state of all metrics to ensure they only reflect the current epoch.
        """
        self.reset_metrics()

    def process_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Perform a single `stage` step on a batch of data from the `stage` set.

        :param batch: A batch of data (a dict) containing the input tensor and target labels.
        :param stage: Either `"train"`, `"val"`, `"test"`, or `"predict"`.
        :return: A tensor of losses between model predictions and targets.
        """
        outputs = self.model.model_step(batch)
        log_dict = {}
        for key, metric in self.model.metrics.items():
            if key not in outputs:
                continue
            value = outputs[key]
            if isinstance(value, (list, tuple)):
                metric.update(*value)
            elif isinstance(value, torch.Tensor):
                metric.update(value)
            else:
                continue
            log_dict[f"{stage}/{key}"] = metric.compute()

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False)
        return outputs["loss"]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        return self.process_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor and target labels.
        :param batch_idx: The index of the current batch.
        """
        self.process_step(batch, stage="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target labels.
        :param batch_idx: The index of the current batch.
        """
        self.process_step(batch, stage="test")

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Any:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a dict) containing the input tensor and target labels.
        :param batch_idx: The index of the current batch.
        :return: The model's prediction for the given batch.
        """
        return self.model.predict_step(batch)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
