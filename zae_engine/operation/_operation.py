from typing import Union, List, Tuple, Optional
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn


class MorphologicalLayer(nn.Module):
    """
    Morphological operation layer for 1D tensors.

    This layer applies a series of morphological operations such as dilation
    and erosion on the input tensor.

    Parameters
    ----------
    ops : str
        A string where each character represents an operation: 'c' for closing
        (dilation followed by erosion) and 'o' for opening (erosion followed by dilation).
    window_size : List[int]
        A list of window sizes for each operation in `ops`.

    Attributes
    ----------
    post : nn.Sequential
        The sequence of morphological operations.
    """

    def __init__(self, ops: str, window_size: List[int]):
        super(MorphologicalLayer, self).__init__()
        try:
            assert len(ops) == len(window_size)
        except AssertionError:
            print("The lengths of the arguments must match.")

        class MorphLayer(nn.Module):
            def __init__(self, kernel_size, morph_type):
                super(MorphLayer, self).__init__()

                self.morph_type = morph_type

                self.conv = nn.Conv1d(1, kernel_size, kernel_size, bias=False, padding="same")
                kernel = torch.zeros((kernel_size, 1, kernel_size), dtype=torch.float)
                for i in range(kernel_size):
                    kernel[i][0][i] = 1
                self.conv.weight.data = kernel

            def forward(self, x):
                """
                Apply morphological operations to the input tensor.

                Parameters
                ----------
                x : torch.Tensor
                    The input tensor.

                Returns
                -------
                torch.Tensor
                    The tensor after morphological operations.
                """
                x = self.conv(x)
                if self.morph_type == "erosion":
                    return torch.min(x, 1)[0].unsqueeze(1)
                elif self.morph_type == "dilation":
                    return torch.max(x, 1)[0].unsqueeze(1)

        morph_list = []
        for op, ker in zip(ops, window_size):
            if op.lower() == "c":
                morph_list += [MorphLayer(int(ker), "dilation"), MorphLayer(int(ker), "erosion")]
            elif op.lower() == "o":
                morph_list += [MorphLayer(int(ker), "erosion"), MorphLayer(int(ker), "dilation")]
            else:
                print("Unexpected operation keyword.")
        self.post = nn.Sequential(*morph_list)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        temp = []
        for i in range(x.shape[1]):
            temp.append(self.post(x[:, i, :].unsqueeze(1)))

        return torch.concat(temp, dim=1)


def label_to_onoff(
    labels: Union[np.ndarray, torch.Tensor], sense: int = 2, middle_only: bool = False, outside_idx: Optional = True
) -> list:
    """
    Convert label sequence to on-off array.

    This function receives the label sequence and returns an on-off array.
    The on-off array consists of [on, off, label] for every existing upper-step and lower-step.
    If there is no label changing and only single label is in input, it returns an empty list.

    Parameters
    ----------
    labels : Union[np.ndarray, torch.Tensor]
        Sequence of annotation for each point. Expected shape is [N, points] or [points] where N is number of data.
    sense : int
        The sensitivity value. Ignore on-off if the (off - on) is less than sensitivity value.
    middle_only : bool
        Ignore both the left-most & right-most on-off.
    outside_idx : Optional[int or float]
        Outside index (default is np.nan). Fill on (or off) if beat is incomplete. Only use for left-most or right-most.
        If middle_only is False, outside_idx is not used.

    Returns
    -------
    list
        On-off matrix: Shape of matrix is (N, # of on-offs, 3) or (# of on-offs, 3) where N is number of data.
        Length of last dimension is 3 and consists of [on, off, label].
    """
    SINGLE = False
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().numpy()
    if not len(labels.shape):
        raise IndexError("Receive empty array.")
    elif len(labels.shape) == 1:
        SINGLE = True
        labels = np.expand_dims(labels.copy(), 0)
    elif len(labels.shape) > 3:
        raise IndexError("Unexpected shape error.")
    else:
        assert len(labels.shape) == 2

    result = []
    for label in labels:
        cursor, res = 0, []
        n_groups = len(list(groupby(label)))
        groups = groupby(label)
        for i, (cls, g) in enumerate(groups):
            g_length = len(list(g))
            if cls:
                if i == 0:
                    if middle_only:
                        pass
                    else:
                        out_start = np.nan if outside_idx else 0
                        res.append([out_start, cursor + g_length - 1, int(cls)])
                elif i == n_groups - 1:
                    if middle_only:
                        pass
                    else:
                        out_end = np.nan if outside_idx else len(label) - 1
                        res.append([cursor, out_end, int(cls)])
                else:
                    if g_length < sense:
                        pass  # not enough length
                    else:
                        res.append([cursor, cursor + g_length - 1, int(cls)])
            else:
                pass  # class #0 is out of interest
            cursor += g_length
        if SINGLE:
            return res
        result.append(res)
    return result


def onoff_to_label(onoff: Union[np.ndarray, torch.Tensor], length: int = 2500) -> np.ndarray:
    """
    Convert on-off array to label sequence.

    This function receives an on-off array and returns the label sequence.
    The on-off array consists of [on, off, label] for existing on-off pairs.
    If there is no beat, it returns an empty array.

    Parameters
    ----------
    onoff : Union[np.ndarray, torch.Tensor]
        Array of on-off. Expected shape is [N, [on, off, cls]] where N is number of on-off pairs.
    length : int
        Length of label sequence. This value should be larger than maximum of onoff.

    Returns
    -------
    np.ndarray
        The label sequence.
    """
    if isinstance(onoff, torch.Tensor):
        onoff = onoff.detach().numpy()
    label = np.zeros(length, dtype=int)
    if len(onoff.shape) == 1:
        return label
    elif len(onoff.shape) > 3:
        raise IndexError("Unexpected shape error.")
    else:
        assert len(onoff.shape) == 2
    if onoff.shape[-1] != 3:
        raise ValueError("Unexpected shape error.")

    for on, off, cls in onoff:
        on = 0 if np.isnan(on) else int(on)
        if np.isnan(off) or (int(off) >= length):
            label[on:] = cls
        else:
            label[on : int(off) + 1] = cls

    return label


def find_nearest(arr: Union[np.ndarray, torch.Tensor], value: int):
    """
    Find the nearest value and its index in the array.

    Parameters
    ----------
    arr : Union[np.ndarray, torch.Tensor]
        The input array.
    value : int
        The reference value.

    Returns
    -------
    Tuple[int, int]
        The index and value of the nearest element.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()

    i_gap = np.searchsorted(arr, value)

    if i_gap == 0:
        return i_gap, arr[0]  # arr의 최소값보다 작은 value
    elif i_gap == len(arr):
        return len(arr) - 1, arr[-1]  # arr의 최대값보다 큰 value
    else:
        left, right = arr[i_gap - 1], arr[i_gap]
        if abs(value - left) <= abs(right - value):
            return i_gap - 1, left
        else:
            return i_gap, right
