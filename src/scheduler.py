# Author: Chanho Kim <theveryrio@gmail.com>

from typing import List

import torch


class DefaultLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """A learning rate scheduler that maintains the initial learning rate throughout training."""

    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        """
        :param optimizer: The optimizer to use for training.
        :param last_epoch: The index of the last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Returns a list of learning rates for each parameter group.

        :return: A list of base learning rates for each parameter group.
        """
        return [base_lr for base_lr in self.base_lrs]
