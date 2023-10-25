from typing import Union, List, Tuple, Optional
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.table import Table


class MorphologicalLayer(nn.Module):
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
    Convert label sequence to onoff array.
    Receive the label(sequence of annotation for each point), return the on-off array.
    On-off array consists of [on, off, class] for exist beats. If there is no beat, return [].

    Input args:
        label: np.nd-array
                Sequence of annotation for each point
                Expected shape is [N, points] or [points] where N is number of data.
        sense: int
                The sensitivity value.
                Ignore beat if the (off - on) is less than sensitivity value.
        middle_only: bool
                Ignore both the left-most & right-most beats.
        outside_idx: int or float(nan)
                Outside index (default is np.nan).
                Fill on (or off) if beat is incomplete. only use for left-most or right-most.
                If middle_only is False, outside_idx is not used.

    Output args:
        Beat info matrix:
                Shape of matrix is (N, # of beats, 3) or (# of beats, 3) where N is number of data.
                Length of last dimension is 3 consists of [on, off, cls].
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
    Return label sequence using onoff(arg #1).
    Receive the label(sequence of annotation for each point), return the on-off array.
    On-off array consists of [on, off, class] for exist beats. If there is no beat, return [].

    :param onoff: np.nd-array. Array of on-off. Expected shape is [N, [on, off, cls]] where N is number of beats.
    :param length: int. Length of label sequence. This value should be larger than maximum of onoff.
    :return: label
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
    Find the nearest value and its index.
    :param arr: 1d-array.
    :param value: reference value.
    :return: index of nearest, value of nearest
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


def draw_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor], y_hat: Union[np.ndarray, torch.Tensor], num_classes: int
):
    """
    Compute confusion matrix.
    Both the y_true and y_hat have data type as integer, and match in shape.

    :param y_true: Union[np.nd-array, torch.Tensor]
    :param y_hat: Union[np.nd-array, torch.Tensor]
    :param num_classes: int
    :return: confusion matrix with 2-D nd-array.
    """

    assert len(y_true) == len(y_hat), f"length unmatched: arg #1 {len(y_true)} =/= arg #2 {len(y_hat)}"
    canvas = np.zeros((num_classes, num_classes))

    for true, hat in zip(y_true, y_hat):
        canvas[true, hat] += 1

    return canvas


def print_confusion_matrix(
    confusion_matrix: np.ndarray,
    cell_width: Optional[int] = 4,
    class_name: Union[List, Tuple] = None,
    frame: Optional[bool] = True,
):
    """
    Printing given confusion matrix.
    Printing width is customizable with cell_width, but height is not.
    The names of rows and columns are customizable with class_name.
    Note that the length of class_name must be matched with the length of the confusion matrix.
    :param confusion_matrix: np.nd-array
    :param cell_width: int, optional
    :param class_name: Union[List[str], Tuple[str]], optional
    :param frame: bool, optional
    :return:
    """
    box_frame = box.SIMPLE if frame else None

    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box_frame,
        leading=1,
        show_edge=True,
        show_lines=True,
        min_width=64,
    )

    console = Console()
    table.title = "\n confusion_matrix"

    if class_name is not None:
        assert (
            len(class_name) == confusion_matrix.shape[-1]
        ), f"Unmatched classes number class_name {len(class_name)} =/= number of class {confusion_matrix.shape[-1]}"

        class_name = [""] + class_name
        for i, name in enumerate(class_name):
            if i == 0:
                table.add_column("", justify="center", style="green", min_width=cell_width, max_width=cell_width)
            else:
                table.add_column(name, justify="center", min_width=cell_width, max_width=cell_width)

        for i, row in enumerate(confusion_matrix):
            row_with_index = [class_name[i + 1]] + list(map(lambda x: str(int(x)), row.tolist()))
            row_with_index[i + 1] = f"[bold cyan]{row_with_index[i + 1]}[bold cyan]"
            table.add_row(*row_with_index)

    else:
        for col in range(confusion_matrix.shape[-1] + 1):
            if col == 0:
                table.add_column("", justify="center", style="green", min_width=cell_width, max_width=cell_width)
            else:
                table.add_column("P" + str(col - 1), justify="center", min_width=cell_width, max_width=cell_width)

        for i, row in enumerate(confusion_matrix):
            row_with_index = [f"T{i}"] + list(map(lambda x: str(int(x)), row.tolist()))
            row_with_index[i + 1] = f"[bold cyan]{row_with_index[i + 1]}[bold cyan]"
            table.add_row(*row_with_index)

    table.caption = "row : [green]Actual[/green] column : [purple]Prediction[/purple]"

    console.print(table)
