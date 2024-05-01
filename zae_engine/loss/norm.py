import torch

from ..metrics import mse


def mse(true: torch.Tensor, predict: torch.Tensor):
    return mse(true, predict)
