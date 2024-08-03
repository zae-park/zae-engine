from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.table import Table

from ..utils.decorators import np2torch, shape_check


@np2torch(dtype=torch.int)
@shape_check(2)
def confusion_matrix(
    y_hat: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor], num_classes: int
) -> torch.Tensor:
    """
    Compute the confusion matrix for classification predictions.

    This function calculates the confusion matrix, comparing the predicted labels (y_hat) with the true labels (y_true).

    Parameters
    ----------
    y_hat : Union[np.ndarray, torch.Tensor]
        The predicted labels, either as a numpy array or a torch tensor.
    y_true : Union[np.ndarray, torch.Tensor]
        The true labels, either as a numpy array or a torch tensor.
    num_classes : int
        The number of classes in the classification task.

    Returns
    -------
    torch.Tensor
        The confusion matrix as a 2-D tensor of shape (num_classes, num_classes).

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 2, 1])
    >>> y_hat = np.array([0, 2, 2, 2, 0])
    >>> confusion_matrix(y_hat, y_true, 3)
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 2.]])
    >>> y_true = torch.tensor([0, 1, 2, 2, 1])
    >>> y_hat = torch.tensor([0, 2, 2, 2, 0])
    >>> confusion_matrix(y_hat, y_true, 3)
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 2.]])
    """
    canvas = torch.zeros((num_classes, num_classes))

    for true, hat in zip(y_true, y_hat):
        canvas[true, hat] += 1

    return canvas


def print_confusion_matrix(
    conf_mat: np.ndarray | torch.Tensor,
    cell_width: Optional[int] = 4,
    class_name: List[str] | Tuple[str] = None,
    frame: Optional[bool] = True,
):
    """
    Print the confusion matrix in a formatted table.

    This function prints the given confusion matrix using the rich library for better visualization. The cell width and class names are customizable.

    Parameters
    ----------
    conf_mat : Union[np.ndarray, torch.Tensor]
        The confusion matrix, either as a numpy array or a torch tensor.
    cell_width : Optional[int], default=4
        The width of each cell in the printed table.
    class_name : Union[List[str], Tuple[str]], optional
        The names of the classes. If provided, must match the number of classes in the confusion matrix.
    frame : Optional[bool], default=True
        Whether to include a frame around the table.

    Returns
    -------
    None

    Examples
    --------
    >>> conf_mat = np.array([[5, 2], [1, 3]])
    >>> print_confusion_matrix(conf_mat, class_name=['Class 0', 'Class 1'])
    >>> conf_mat = torch.tensor([[5, 2], [1, 3]])
    >>> print_confusion_matrix(conf_mat, class_name=['Class 0', 'Class 1'])
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
            len(class_name) == conf_mat.shape[-1]
        ), f"Unmatched classes number class_name {len(class_name)} =/= number of class {conf_mat.shape[-1]}"

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
        for col in range(conf_mat.shape[-1] + 1):
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
