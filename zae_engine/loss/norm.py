import torch

from ..metrics import mse as _mse


def mse(true: torch.Tensor, predict: torch.Tensor):
    """
    Compute the mean squared error (MSE) between the true and predicted values.

    Parameters
    ----------
    true : torch.Tensor
        The ground truth values.
    predict : torch.Tensor
        The predicted values.

    Returns
    -------
    torch.Tensor
        The computed mean squared error.
    """
    return _mse(true, predict)