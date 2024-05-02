from typing import Union, Optional, Tuple, List, Any

import numpy as np
import torch

from ..operation import draw_confusion_matrix
from ..utils.deco import EPS, np2torch, shape_check


@np2torch(dtype=torch.int)
@shape_check
def accuracy(
    true: Union[np.ndarray, torch.Tensor],
    predict: Union[np.ndarray, torch.Tensor],
):
    correct = torch.eq(true, predict)
    return sum(correct) / len(correct)


def fbeta(*args, beta: float, num_classes: int, average: str = "micro"):
    """
    Compute f-beta score using given confusion matrix (args#1 with asterisk).
    If the first argument is a tuple of length 2, i.e. true and prediction, then compute confusion matrix first.
    Support two average methods to calculate precision or recall, i.e. micro- and macro-.

    :param args:
        Confusion matrix (or true and prediction)
    :param beta: float
    :param num_classes: int
    :param average: str
        If 'micro', precision and recall are derived using TP and FP for all classes.
        If 'macro', precision and recall are derived using precision and recall for each class.
    """

    if len(args) == 2:
        conf = draw_confusion_matrix(*args, num_classes=num_classes)
    else:
        conf = args[0]

    tp_set = conf.diagonal()
    row = conf.sum(1)
    col = conf.sum(0)

    if average == "micro":
        micro_tp = tp_set.sum()
        micro_fn = (row - tp_set).sum()
        micro_fp = (col - tp_set).sum()

        recall = micro_tp / (micro_tp + micro_fn + eps)
        precision = micro_tp / (micro_tp + micro_fp + eps)

        micro_f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + EPS)
        return micro_f1

    elif average == "macro":
        macro_f1 = 0
        for tp, r, c in zip(tp_set, row, col):
            precision = tp / (c + eps)
            recall = tp / (r + eps)
            f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + EPS)
            macro_f1 += f1

        return macro_f1 / num_classes

