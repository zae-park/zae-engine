from typing import Union

from typeguard import typechecked
import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import giou, miou


@typechecked
def mIoU(labels: Union[np.ndarray, torch.Tensor], outputs: Union[np.ndarray, torch.Tensor]):
    """
    Compute mean IoU using given outputs and labels.
    The outputs and labels are 1-D or 2-D array.
    :param outputs: Union[np.ndarray, torch.Tensor] -> elements of array must be int or bool type
    :param labels: Union[np.ndarray, torch.Tensor] -> elements of array must be int or bool type
    """
    assert (
        "int" in str(outputs.dtype).lower() or "bool" in str(outputs.dtype).lower()
    ), f"outputs array's elements data type must be int or bool type current element type is {outputs.dtype}"

    assert (
        "int" in str(labels.dtype) or "bool" in str(labels.dtype).lower()
    ), f"labels array's elements data type must be int or bool type current element type is {labels.dtype}"

    score = miou(img1=outputs, img2=labels)
    return torch.mean(score)


@typechecked
def IoU(true_onoff: Union[np.ndarray, torch.Tensor], pred_onoff: Union[np.ndarray, torch.Tensor]):
    """
    Compute mean IoU using given true_onoff and pred_onoff.
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

    _, iou = giou(true_onoff=true_onoff, pred_onoff=pred_onoff, iou=True)
    return torch.mean(1 - iou)


@typechecked
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

    score = giou(true_onoff=true_onoff, pred_onoff=pred_onoff)
    return torch.mean(1 - score)
