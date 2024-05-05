import torch

from ..metrics import mse as _mse


def mse(true: torch.Tensor, predict: torch.Tensor):
    return _mse(true, predict)
