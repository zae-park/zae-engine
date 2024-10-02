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


def arg_nearest(
    arr: Union[np.ndarray, torch.Tensor], value: int, return_value: bool = True
) -> Union[int, Tuple[int, int]]:
    """
    Find the index of the nearest value in the array to the given reference value.
    Optionally, also return the nearest value itself.

    Parameters
    ----------
    arr : Union[np.ndarray, torch.Tensor]
        The input sorted array. Must be in ascending order.
    value : int
        The reference value to find the nearest element for.
    return_value : bool, optional
        Whether to return the nearest value along with its index.
        - If `True`, returns a tuple `(index, nearest_value)`.
        - If `False`, returns only the `index` of the nearest value.
        Default is `True`.

    Returns
    -------
    Union[int, Tuple[int, int]]
        - If `return_value` is `True`, returns a tuple containing:
            - `index` (int): The index of the nearest element in the array.
            - `nearest_value` (int): The nearest value to the reference `value`.
        - If `return_value` is `False`, returns only:
            - `index` (int): The index of the nearest element in the array.

    Raises
    ------
    ValueError
        If `arr` is not a one-dimensional sorted array.
    TypeError
        If `arr` is neither a NumPy array nor a PyTorch tensor.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 3, 5, 7, 9])
    >>> arg_nearest(arr, 6)
    (2, 5)

    >>> arg_nearest(arr, 6, return_value=False)
    2

    >>> import torch
    >>> arr_tensor = torch.tensor([2, 4, 6, 8, 10])
    >>> arg_nearest(arr_tensor, 7)
    (2, 6)

    >>> arg_nearest(arr_tensor, 7, return_value=False)
    2
    """
    # Check if the input array is a PyTorch tensor and convert it to a NumPy array
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    elif not isinstance(arr, np.ndarray):
        raise TypeError("Input array must be a NumPy array or a PyTorch tensor.")

    # Ensure the array is one-dimensional
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    # Check if the array is sorted in ascending order
    if not np.all(arr[:-1] <= arr[1:]):
        raise ValueError("Input array must be sorted in ascending order.")

    # Use np.searchsorted to find the insertion index for the reference value
    i_gap = np.searchsorted(arr, value)

    # Handle boundary conditions
    if i_gap == 0:
        nearest_index = 0
        nearest_value = arr[0]  # Value is smaller than the smallest element in the array
    elif i_gap == len(arr):
        nearest_index = len(arr) - 1
        nearest_value = arr[-1]  # Value is larger than the largest element in the array
    else:
        left, right = arr[i_gap - 1], arr[i_gap]
        # Determine which of the two neighboring values is closer to the reference value
        if abs(value - left) <= abs(right - value):
            nearest_index = i_gap - 1
            nearest_value = left
        else:
            nearest_index = i_gap
            nearest_value = right

    # Return the result based on the return_value flag
    if return_value:
        return nearest_index, nearest_value
    else:
        return nearest_index
