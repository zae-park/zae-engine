from typing import Union, Tuple, List

import numpy as np
import torch

from ..utils import deco


@deco.np2torch(dtype=torch.int)
@deco.shape_check(2)
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
        iou_ += intersection / (union + deco.EPS)

    return iou_ / maximum


@deco.np2torch(dtype=torch.int)
@deco.shape_check('img1', 'img2')
def giou(
    img1: np.ndarray | torch.Tensor, img2: np.ndarray | torch.Tensor,
    iou: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compute mean GIoU and IoU for given outputs and labels.
    :param img1: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param img2: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param iou: if True, return IoU with GIoU. Default is False.
    :return: GIoU, iou (option) with shape [-1].
    """

    if len(img1.shape) == 1:
        img1 = img1.clone().unsqueeze(0)
    if len(img2.shape) == 1:
        img2 = img2.clone().unsqueeze(0)

    true_on, true_off = img1[:, 0], img1[:, 1]
    pred_on, pred_off = img2[:, 0], img2[:, 1]
    C_on = torch.min(true_on, pred_on)
    C_off = torch.max(true_off, pred_off)

    C_area = C_off - C_on
    relative_area = C_area - (true_off - true_on)
    union = C_area  # they are same in 1-dimension
    intersection = torch.min(abs(true_on - pred_off), abs(true_off - pred_on))

    IoU = intersection / (union + deco.EPS)
    if iou:
        return IoU - abs(relative_area / (C_area + deco.EPS)), IoU
    else:
        return IoU - abs(relative_area / (C_area + deco.EPS))