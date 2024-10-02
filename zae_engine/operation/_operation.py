from typing import Union, List, Tuple

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
        return nearest_index, int(nearest_value)
    else:
        return nearest_index
