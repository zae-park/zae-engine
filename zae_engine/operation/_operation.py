from typing import Union, List, Tuple, Optional
from itertools import groupby
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Run:
    start_index: int
    end_index: int
    value: int
    # is_kept: bool


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
    labels: Union[np.ndarray, torch.Tensor],
    sense: int = 2,
    middle_only: bool = False,
    outside_idx: Optional[float] = np.nan,
) -> list:
    """
    Convert label sequence to Run-Length Encoding (RLE) on-off runs.

    This function takes a label sequence and encodes it into a list of runs.
    Each run consists of [start, end, label], representing the start and end indices
    of consecutive occurrences of a specific label. Runs where the label does not change
    or the run length is below the specified sensitivity are ignored.

    Parameters
    ----------
    labels : Union[np.ndarray, torch.Tensor]
        Sequence of labels. Expected shape is [N, points] or [points] where N is the number of data samples.
    sense : int, optional
        The minimum run length to consider. Runs shorter than this value are ignored. Default is 2.
    middle_only : bool, optional
        If True, ignores the first and last runs in the sequence. Default is False.
    outside_idx : Optional[float], optional
        Value to use for the start of the first run and the end of the last run if they are incomplete.
        If set to `np.nan`, it indicates an undefined boundary. Only used when `middle_only` is False.
        Default is `np.nan`.

    Returns
    -------
    list
        On-off runs encoded as a list of lists. Each inner list contains [start, end, label].
        Shape is (N, # of runs, 3) if input is multi-dimensional, or (# of runs, 3) for single data.
    """
    SINGLE = False
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().numpy()
    if not len(labels.shape):
        raise IndexError("Received an empty array.")
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
        groups = list(groupby(label))
        n_groups = len(groups)
        for i, (cls, g) in enumerate(groups):
            g_length = len(list(g))
            if cls != 0:  # Assuming label 0 is background and not of interest
                if i == 0:
                    if not middle_only:
                        start = outside_idx if np.isnan(outside_idx) else 0
                        end = cursor + g_length - 1
                        res.append([start, end, int(cls)])
                elif i == n_groups - 1:
                    if not middle_only:
                        start = cursor
                        end = outside_idx if np.isnan(outside_idx) else len(label) - 1
                        res.append([start, end, int(cls)])
                else:
                    if g_length >= sense:
                        start = cursor
                        end = cursor + g_length - 1
                        res.append([start, end, int(cls)])
            cursor += g_length
        if SINGLE:
            return res
        result.append(res)
    return result


def onoff_to_label(onoff: Union[np.ndarray, torch.Tensor], length: int = 2500) -> np.ndarray:
    """
    Convert Run-Length Encoding (RLE) on-off runs back to label sequence.

    This function takes a list of RLE on-off runs and reconstructs the original label sequence.
    Each run consists of [start, end, label], indicating that the label is active from the start
    index to the end index (inclusive).

    Parameters
    ----------
    onoff : Union[np.ndarray, torch.Tensor]
        Array of RLE runs. Expected shape is [N, 3], where each row represents [start, end, label].
    length : int, optional
        Length of the output label sequence. This value should be greater than the maximum end index.
        Default is 2500.

    Returns
    -------
    np.ndarray
        The reconstructed label sequence of shape [length].
    """
    if isinstance(onoff, torch.Tensor):
        onoff = onoff.detach().numpy()
    label = np.zeros(length, dtype=int)
    if len(onoff.shape) == 1:
        return label
    elif len(onoff.shape) > 2:
        raise IndexError("Unexpected shape error.")
    else:
        assert len(onoff.shape) == 2
    if onoff.shape[-1] != 3:
        raise ValueError("Last dimension of onoff must be 3.")

    for run in onoff:
        on, off, cls = run
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
