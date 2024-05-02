from typing import Union

from typeguard import typechecked
import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import giou as _giou, miou as _miou
from ..utils.deco import shape_check


@shape_check(2)
def mIoU(pred: torch.Tensor, true: torch.Tensor):
    """
    Compute mean IoU using given outputs and labels.
    The outputs and labels are 1-D or 2-D array.
    :param pred: torch.Tensor -> elements of array must be int or bool type
    :param true: torch.Tensor -> elements of array must be int or bool type
    """

    score = _miou(img1=pred, img2=true)
    return torch.mean(score)


@shape_check(2)
def IoU(pred: torch.Tensor, true: torch.Tensor):
    """
    Compute mean IoU using given true_onoff and pred_onoff.
    The true_onoff and pred_onoff are 2-D array.

    :param pred: Union[np.ndarray, torch.Tensor]  -> elements of array must be int
    :param true: Union[np.ndarray, torch.Tensor]  -> elements of array must be int
    """

    _, iou = _giou(img1=pred, img2=true, iou=True)
    return torch.mean(1 - iou)


@shape_check(2)
def GIoU(true_onoff: Union[np.ndarray, torch.Tensor], pred_onoff: Union[np.ndarray, torch.Tensor]):
    """
    Compute mean generalized IoU using given true_onoff and pred_onoff (https://arxiv.org/abs/1902.09630v2).
    The true_onoff and pred_onoff are 2-D array.

    :param true_onoff: Union[np.ndarray, torch.Tensor]  -> elements of array must be int
    :param pred_onoff: Union[np.ndarray, torch.Tensor]  -> elements of array must be int
    """

    assert (
        "int" in str(true_onoff.dtype).lower()
    ), f"true_onoff array's elements data type must be int, but receive {true_onoff.dtype}"

    assert (
        "int" in str(pred_onoff.dtype).lower()
    ), f"pred_onoff array's elements data type must be int, but receive {pred_onoff.dtype}"

    score = _giou(true_onoff=true_onoff, pred_onoff=pred_onoff)
    return torch.mean(1 - score)
