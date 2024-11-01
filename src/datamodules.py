# Author: Chanho Kim <theveryrio@gmail.com>

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class CSVDataset(Dataset):
    """A dataset class for loading data from a CSV file."""

    def __init__(
        self,
        csv_path: Union[str, Path] = "",
        x_columns: List[Union[int, str]] = [],
        y_columns: List[Union[int, str]] = [],
        transform: Optional[Callable] = None,
    ) -> None:
        """
        :param csv_path: The path to the CSV file. Defaults to `""`.
        :param x_columns: List of column names or indices to be used as features. Defaults to `[]`.
        :param y_columns: List of column names or indices to be used as labels. Defaults to `[]`.
        :param transform: The callable object for preprocessing. Defaults to `None`.
        """
        super().__init__()

        base, ext = os.path.splitext(csv_path)
        if ext.lower() != ".csv":
            csv_path = base + ".csv"

        if not os.path.exists(csv_path):
            return
        if os.path.isdir(csv_path):
            return

        columns = pd.read_csv(csv_path, nrows=0).columns.to_list()
        self.x_columns = [columns[col] if isinstance(col, int) else col for col in x_columns]
        self.y_columns = [columns[col] if isinstance(col, int) else col for col in y_columns]
        self.df = pd.read_csv(csv_path, usecols=self.x_columns + self.y_columns)

        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        :return: The number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves a sample from the dataset at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A dictionary containing the sample data, where keys are prefixed with 'x' for features
            and 'y' for labels, corresponding to the columns specified in x_columns and y_columns.
            The dictionary structure is as follows:
            - Keys with 'x' prefix represent feature values.
            - Keys with 'y' prefix represent label values.
        """
        output_dict = self.df.iloc[idx].to_dict()

        if self.transform:
            output_dict = self.transform(output_dict)

        for i, col in enumerate(self.x_columns):
            output_dict[f"x{i}"] = output_dict.pop(col)
        for i, col in enumerate(self.y_columns):
            output_dict[f"y{i}"] = output_dict.pop(col)

        return output_dict


class TemporarySplitDataModule(LightningDataModule):
    """A data module for a temporarily split dataset."""

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        :param train_dataset: The `Dataset` class for train.
        :param val_dataset: The `Dataset` class for val.
        :param test_dataset: The `Dataset` class for test.
        :param predict_dataset: The `Dataset` class for predict.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.predict_data: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.num_workers = os.cpu_count() // 2
        self.pin_memory = True

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_data`, `self.val_data`, `self.test_data`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.train_data and hasattr(self.hparams.train_dataset, "df"):
            self.train_data = self.hparams.train_dataset
        if not self.val_data and hasattr(self.hparams.val_dataset, "df"):
            self.val_data = self.hparams.val_dataset
        if not self.test_data and hasattr(self.hparams.test_dataset, "df"):
            self.test_data = self.hparams.test_dataset
        if not self.predict_data and hasattr(self.hparams.predict_dataset, "df"):
            self.predict_data = self.hparams.predict_dataset

    def train_dataloader(self) -> Optional[DataLoader[Any]]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if not self.train_data:
            return None
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> Optional[DataLoader[Any]]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if not self.val_data:
            return None
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader[Any]]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if not self.test_data:
            return None
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> Optional[DataLoader[Any]]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        if not self.predict_data:
            return None
        return DataLoader(
            dataset=self.predict_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
