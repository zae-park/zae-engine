from typing import Union

import numpy as np
import torch

from ..metrics import giou as _giou, miou as _miou
from ..utils.decorators import shape_check


@shape_check(2)
def mIoU(pred: torch.Tensor, true: torch.Tensor):
    """
    Compute mean Intersection over Union (mIoU) using the given predicted and true labels.

    The outputs and labels must be 1-D or 2-D tensors with elements of integer or boolean type.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels tensor. Elements must be of type int or bool.
    true : torch.Tensor
        True labels tensor. Elements must be of type int or bool.

    Returns
    -------
    torch.Tensor
        The mean IoU score.
    """
    score = _miou(img1=pred, img2=true)
    return torch.mean(score)


@shape_check(2)
def IoU(pred: torch.Tensor, true: torch.Tensor):
    """
    Compute mean Intersection over Union (IoU) using the given true and predicted labels.

    The true and predicted labels must be 2-D tensors with elements of integer type.

    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Predicted labels tensor. Elements must be of type int.
    true : Union[np.ndarray, torch.Tensor]
        True labels tensor. Elements must be of type int.

    Returns
    -------
    torch.Tensor
        The mean IoU score.
    """
    _, iou = _giou(img1=pred, img2=true, iou=True)
    return torch.mean(1 - iou)


@shape_check(2)
def GIoU(true_onoff: Union[np.ndarray, torch.Tensor], pred_onoff: Union[np.ndarray, torch.Tensor]):
    """
    Compute mean Generalized Intersection over Union (GIoU) using the given true and predicted labels.

    The true and predicted labels must be 2-D tensors with elements of integer type.
    See https://arxiv.org/abs/1902.09630v2 for details on GIoU.

    Parameters
    ----------
    true_onoff : Union[np.ndarray, torch.Tensor]
        True labels tensor. Elements must be of type int.
    pred_onoff : Union[np.ndarray, torch.Tensor]
        Predicted labels tensor. Elements must be of type int.

    Returns
    -------
    torch.Tensor
        The mean GIoU score.

    Raises
    ------
    AssertionError
        If the elements of true_onoff or pred_onoff are not of integer type.
    """
    assert (
        "int" in str(true_onoff.dtype).lower()
    ), f"true_onoff array's elements data type must be int, but received {true_onoff.dtype}"

    assert (
        "int" in str(pred_onoff.dtype).lower()
    ), f"pred_onoff array's elements data type must be int, but received {pred_onoff.dtype}"

    score = _giou(true_onoff=true_onoff, pred_onoff=pred_onoff)
    return torch.mean(1 - score)
