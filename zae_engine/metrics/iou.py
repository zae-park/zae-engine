from typing import Union, Optional, List

import numpy as np
import torch

from .utils import np2torch


@np2torch
def miou(
    outputs: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """
    Compute mean IoU for given outputs and labels.
    :param outputs: Shape - [-1, dim]. tensor (or nd-array) of model's outputs.
    :param labels: Shape - [-1, dim]. tensor (or nd-array) of labels.
    :return: mIoU with shape [-1].
    """

    assert (
        "int" in str(outputs.dtype).lower() or "bool" in str(outputs.dtype).lower()
    ), f"outputs array's elements data type must be int or bool type current element type is {outputs.dtype}"

    assert (
        "int" in str(labels.dtype) or "bool" in str(labels.dtype).lower()
    ), f"labels array's elements data type must be int or bool type current element type is {labels.dtype}"

    assert outputs.shape == labels.shape, f"Shape unmatched: arg #1 {outputs.shape} =/= arg #2 {labels.shape}"
    if isinstance(outputs, np.ndarray):
        outputs = torch.tensor(outputs)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    if len(labels.shape) == 1:
        labels = labels.clone().reshape(1, -1)
        outputs = outputs.clone().reshape(1, -1)
    n = len(labels)

    maximum = int(max(outputs.max(), labels.max()))
    iou_ = torch.zeros(n)
    for m in range(maximum):
        intersection = ((outputs == m).int() & (labels == m).int()).float().sum(-1)
        union = ((outputs == m).int() | (labels == m).int()).float().sum(-1)
        iou_ += intersection / (union + torch.finfo(torch.float32).eps)

    return iou_ / maximum


@np2torch
def giou(
    true_onoff: Union[np.ndarray, torch.Tensor, List[Union[int]]],
    pred_onoff: Union[np.ndarray, torch.Tensor, List[Union[int]]],
    iou: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compute mean GIoU and IoU for given outputs and labels.
    :param true_onoff: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param pred_onoff: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param iou: if True, return IoU with GIoU. Default is False.
    :return: GIoU, iou (option) with shape [-1].
    """

    if not isinstance(true_onoff, torch.Tensor):
        true_onoff = torch.tensor(true_onoff)
    if not isinstance(pred_onoff, torch.Tensor):
        pred_onoff = torch.tensor(pred_onoff)

    assert (
        "int" in str(true_onoff.dtype).lower()
    ), f"true_onoff array's elements data type must be int, but receive {true_onoff.dtype}"

    assert (
        "int" in str(pred_onoff.dtype).lower()
    ), f"pred_onoff array's elements data type must be int, but receive {pred_onoff.dtype}"

    if len(true_onoff.shape) == 1:
        true_onoff = true_onoff.clone().unsqueeze(0)
    if len(pred_onoff.shape) == 1:
        pred_onoff = pred_onoff.clone().unsqueeze(0)
    assert (
        true_onoff.shape == pred_onoff.shape
    ), f"Shape unmatched: arg #1 {true_onoff.shape} =/= arg #2 {pred_onoff.shape}"

    true_on, true_off = true_onoff[:, 0], true_onoff[:, 1]
    pred_on, pred_off = pred_onoff[:, 0], pred_onoff[:, 1]
    C_on = torch.min(true_on, pred_on)
    C_off = torch.max(true_off, pred_off)

    eps = +torch.finfo(torch.float32).eps
    C_area = C_off - C_on
    relative_area = C_area - (true_off - true_on)
    union = C_area  # they are same in 1-dimension
    intersection = torch.min(abs(true_on - pred_off), abs(true_off - pred_on))

    IoU = intersection / (union + eps)
    if iou:
        return IoU - abs(relative_area / (C_area + eps)), IoU
    else:
        return IoU - abs(relative_area / (C_area + eps))