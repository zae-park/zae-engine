from typing import Union, Tuple, List

import numpy as np
import torch

from ..utils.deco import EPS, np2torch, shape_check


@np2torch(dtype=torch.int)
@shape_check
def miou(img1: np.ndarray | torch.Tensor, img2: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Compute mean IoU for each value in given images(arguments).
    TODO: this function works for 1-dimensional arrays or tensors, 2 or greater dimensional mode will be update.
    :param img1: Shape - [-1, dim]. tensor (or nd-array) of model's outputs.
    :param img2: Shape - [-1, dim]. tensor (or nd-array) of labels.
    :return: mIoU with shape [-1].
    """

    if len(img1.shape) == 1:
        img1 = img1.clone().reshape(1, -1)
        img2 = img2.clone().reshape(1, -1)
    n = len(img1)

    maximum = int(max(img1.max(), img2.max()))
    iou_ = torch.zeros(n)
    for m in range(maximum):
        intersection = ((img1 == m).int() & (img2 == m).int()).float().sum(-1)
        union = ((img1 == m).int() | (img2 == m).int()).float().sum(-1)
        iou_ += intersection / (union + EPS)

    return iou_ / maximum


@np2torch(dtype=torch.int)
@shape_check
def giou(
    true_onoff: Union[np.ndarray, torch.Tensor, List],
    pred_onoff: Union[np.ndarray, torch.Tensor, List],
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