from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.table import Table

from ..utils.deco import shape_check, np2torch


@np2torch(dtype=torch.int)
@shape_check(2)
def confusion_matrix(
    y_hat: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor], num_classes: int
) -> torch.Tensor:
    """
    Compute confusion matrix.
    Both the y_true and y_hat have data type as integer, and match in shape.

    :param y_true: Union[np.nd-array, torch.Tensor]
    :param y_hat: Union[np.nd-array, torch.Tensor]
    :param num_classes: int
    :return: confusion matrix with 2-D nd-array.
    """
    canvas = torch.zeros((num_classes, num_classes))

    for true, hat in zip(y_true, y_hat):
        canvas[true, hat] += 1

    return canvas


@np2torch(dtype=torch.int)
def print_confusion_matrix(
    conf_mat: np.ndarray | torch.Tensor,
    cell_width: Optional[int] = 4,
    class_name: List[str] | Tuple[str] = None,
    frame: Optional[bool] = True,
):
    """
    Printing given confusion matrix.
    Printing width is customizable with cell_width, but height is not.
    The names of rows and columns are customizable with class_name.
    Note that the length of class_name must be matched with the length of the confusion matrix.
    :param conf_mat: np.nd-array or torch.Tensor
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

        for i, row in enumerate(conf_mat):
            row_with_index = [class_name[i + 1]] + list(map(lambda x: str(int(x)), row.tolist()))
            row_with_index[i + 1] = f"[bold cyan]{row_with_index[i + 1]}[bold cyan]"
            table.add_row(*row_with_index)

    else:
        for col in range(confusion_matrix.shape[-1] + 1):
            if col == 0:
                table.add_column("", justify="center", style="green", min_width=cell_width, max_width=cell_width)
            else:
                table.add_column("P" + str(col - 1), justify="center", min_width=cell_width, max_width=cell_width)

        for i, row in enumerate(conf_mat):
            row_with_index = [f"T{i}"] + list(map(lambda x: str(int(x)), row.tolist()))
            row_with_index[i + 1] = f"[bold cyan]{row_with_index[i + 1]}[bold cyan]"
            table.add_row(*row_with_index)

    table.caption = "row : [green]Actual[/green] column : [purple]Prediction[/purple]"

    console.print(table)
