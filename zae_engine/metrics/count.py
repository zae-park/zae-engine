from typing import Union

import numpy as np
import torch

from . import confusion
from ..utils import deco


@deco.np2torch(dtype=torch.int)
@deco.shape_check(2)
def accuracy(
    true: Union[np.ndarray, torch.Tensor],
    predict: Union[np.ndarray, torch.Tensor],
):
    """
    Compute the accuracy of predictions.

    This function compares the true labels with the predicted labels and calculates the accuracy.

    Parameters
    ----------
    true : Union[np.ndarray, torch.Tensor]
        The true labels, either as a numpy array or a torch tensor. Shape should be [-1, dim].
    predict : Union[np.ndarray, torch.Tensor]
        The predicted labels, either as a numpy array or a torch tensor. Shape should be [-1, dim].

    Returns
    -------
    torch.Tensor
        The accuracy of the predictions as a torch tensor.

    Examples
    --------
    >>> true = np.array([1, 2, 3, 4])
    >>> predict = np.array([1, 2, 2, 4])
    >>> accuracy(true, predict)
    tensor(0.7500)
    >>> true = torch.tensor([1, 2, 3, 4])
    >>> predict = torch.tensor([1, 2, 2, 4])
    >>> accuracy(true, predict)
    tensor(0.7500)
    """
    correct = torch.eq(true, predict)
    return sum(correct) / len(correct)


@deco.np2torch(dtype=torch.int)
@deco.shape_check(2)
def f_beta(
    pred: np.ndarray | torch.Tensor,
    true: np.ndarray | torch.Tensor,
    beta: float,
    num_classes: int,
    average: str = "micro",
):
    """
    Compute f-beta score using given confusion matrix (args#1 with asterisk).
    If the first argument is a tuple of length 2, i.e. true and prediction, then compute confusion matrix first.
    Support two average methods to calculate precision or recall, i.e. micro- and macro-.

    :param pred:
    :param true:
        For compute confusion matrix
    :param beta: float
    :param num_classes: int
    :param average: str
        If 'micro', precision and recall are derived using TP and FP for all classes.
        If 'macro', precision and recall are derived using precision and recall for each class.
    """

    conf = confusion.confusion_matrix(pred, true, num_classes=num_classes)
    return f_beta_from_mat(conf, beta=beta, num_classes=num_classes, average=average)


@deco.np2torch(dtype=torch.int)
def f_beta_from_mat(conf_mat: np.ndarray | torch.Tensor, beta: float, num_classes: int, average: str = "micro"):
    """
    Compute f-beta score using given confusion matrix (args#1 with asterisk).
    If the first argument is a tuple of length 2, i.e. true and prediction, then compute confusion matrix first.
    Support two average methods to calculate precision or recall, i.e. micro- and macro-.

    :param conf_mat:
        Confusion matrix
    :param beta: float
    :param num_classes: int
    :param average: str
        If 'micro', precision and recall are derived using TP and FP for all classes.
        If 'macro', precision and recall are derived using precision and recall for each class.
    """

    tp_set = conf_mat.diagonal()
    row = conf_mat.sum(1)
    col = conf_mat.sum(0)

    if average == "micro":
        micro_tp = tp_set.sum()
        micro_fn = (row - tp_set).sum()
        micro_fp = (col - tp_set).sum()

        recall = micro_tp / (micro_tp + micro_fn + deco.EPS)
        precision = micro_tp / (micro_tp + micro_fp + deco.EPS)

        micro_f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + deco.EPS)
        return micro_f1

    elif average == "macro":
        macro_f1 = 0
        for tp, r, c in zip(tp_set, row, col):
            precision = tp / (c + deco.EPS)
            recall = tp / (r + deco.EPS)
            f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + deco.EPS)
            macro_f1 += f1

        return macro_f1 / num_classes
