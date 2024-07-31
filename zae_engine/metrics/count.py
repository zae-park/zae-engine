from typing import Union, Tuple

import numpy as np
import torch

from . import confusion
from ..utils import EPS
from ..utils.decorators import np2torch, shape_check


@np2torch(dtype=torch.int)
@shape_check(2)
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


class Acc:
    """
    Accuracy calculation class.

    This class calculates either top-1 or top-k accuracy based on the input shapes.

    Methods
    -------
    __call__(true, predict):
        Computes the accuracy based on the input shapes.
    """

    def __call__(
        self,
        true: Union[np.ndarray, torch.Tensor],
        predict: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the accuracy of predictions.

        If predict has one more dimension than true, calculates top-k accuracy.
        Otherwise, calculates top-1 accuracy.

        Parameters
        ----------
        true : Union[np.ndarray, torch.Tensor]
            The true labels, either as a numpy array or a torch tensor.
        predict : Union[np.ndarray, torch.Tensor]
            The predicted labels, either as a numpy array or a torch tensor.

        Returns
        -------
        torch.Tensor
            The accuracy of the predictions as a torch tensor.
        """
        true, predict = self._check_inputs(true, predict)

        if predict.dim() == true.dim() + 1:
            k = predict.size(-1)
            return self.top_k_accuracy(true, predict, k)
        elif predict.dim() == true.dim():
            return self.accuracy(true, predict)
        else:
            raise ValueError(
                "The dimensions of 'predict' must be either equal to or one more than the dimensions of 'true'."
            )

    @staticmethod
    @np2torch(dtype=torch.int)
    @shape_check(2)
    def accuracy(
        true: Union[np.ndarray, torch.Tensor],
        predict: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute the top-1 accuracy of predictions.

        This function compares the true labels with the predicted labels and calculates the top-1 accuracy.

        Parameters
        ----------
        true : Union[np.ndarray, torch.Tensor]
            The true labels, either as a numpy array or a torch tensor. Shape should be [-1, dim].
        predict : Union[np.ndarray, torch.Tensor]
            The predicted labels, either as a numpy array or a torch tensor. Shape should be [-1, dim].

        Returns
        -------
        torch.Tensor
            The top-1 accuracy of the predictions as a torch tensor.
        """
        correct = torch.eq(true, predict)
        return torch.mean(correct.float()).item()

    @staticmethod
    @np2torch(dtype=torch.float32)
    def top_k_accuracy(
        true: Union[np.ndarray, torch.Tensor], predict: Union[np.ndarray, torch.Tensor], k: int
    ) -> float:
        """
        Compute the top-k accuracy of predictions.

        This function compares the true labels with the top-k predicted labels and calculates the top-k accuracy.

        Parameters
        ----------
        true : Union[np.ndarray, torch.Tensor]
            The true labels, either as a numpy array or a torch tensor.
        predict : Union[np.ndarray, torch.Tensor]
            The predicted labels, either as a numpy array or a torch tensor.
        k : int
            The number of top predictions to consider for calculating the accuracy.

        Returns
        -------
        torch.Tensor
            The top-k accuracy of the predictions as a torch tensor.
        """
        if predict.dim() == 1:
            predict = predict.unsqueeze(1)

        topk = torch.topk(predict, k, dim=1).indices
        correct = topk.eq(true.view(-1, 1).expand_as(topk))
        return (correct.float().sum() / len(true)).item()

    def _check_inputs(
        self, true: Union[np.ndarray, torch.Tensor], predict: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert inputs to torch tensors and check their shapes.

        Parameters
        ----------
        true : Union[np.ndarray, torch.Tensor]
            The true labels.
        predict : Union[np.ndarray, torch.Tensor]
            The predicted labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The converted true and predict tensors.
        """
        if isinstance(true, np.ndarray):
            true = torch.tensor(true, dtype=torch.int)
        if isinstance(predict, np.ndarray):
            predict = torch.tensor(predict, dtype=torch.float)

        if true.shape[0] != predict.shape[0]:
            raise ValueError("The first dimension of 'true' and 'predict' must be the same.")

        return true, predict


@np2torch(dtype=torch.int, n=2)
@shape_check(2)
def f_beta(
    pred: np.ndarray | torch.Tensor,
    true: np.ndarray | torch.Tensor,
    beta: float,
    num_classes: int,
    average: str = "micro",
):
    """
    Compute the F-beta score.

    This function calculates the F-beta score for the given predictions and true labels, supporting both micro and macro averaging.

    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        The predicted labels, either as a numpy array or a torch tensor.
    true : Union[np.ndarray, torch.Tensor]
        The true labels, either as a numpy array or a torch tensor.
    beta : float
        The beta value for the F-beta score calculation.
    num_classes : int
        The number of classes in the classification task.
    average : str
        The averaging method for the F-beta score calculation. Either 'micro' or 'macro'.

    Returns
    -------
    torch.Tensor
        The F-beta score as a torch tensor.

    Examples
    --------
    >>> pred = np.array([1, 2, 3, 4])
    >>> true = np.array([1, 2, 2, 4])
    >>> f_beta(pred, true, beta=1.0, num_classes=5, average='micro')
    tensor(0.8000)
    >>> pred = torch.tensor([1, 2, 3, 4])
    >>> true = torch.tensor([1, 2, 2, 4])
    >>> f_beta(pred, true, beta=1.0, num_classes=5, average='micro')
    tensor(0.8000)
    """

    conf = confusion.confusion_matrix(pred, true, num_classes=num_classes)
    return f_beta_from_mat(conf, beta=beta, num_classes=num_classes, average=average)


@np2torch(dtype=torch.int, n=1)
def f_beta_from_mat(conf_mat: np.ndarray | torch.Tensor, beta: float, num_classes: int, average: str = "micro"):
    """
    Compute the F-beta score from a given confusion matrix.

    This function calculates the F-beta score using the provided confusion matrix, supporting both micro and macro averaging.

    Parameters
    ----------
    conf_mat : Union[np.ndarray, torch.Tensor]
        The confusion matrix, either as a numpy array or a torch tensor.
    beta : float
        The beta value for the F-beta score calculation.
    num_classes : int
        The number of classes in the classification task.
    average : str
        The averaging method for the F-beta score calculation. Either 'micro' or 'macro'.

    Returns
    -------
    torch.Tensor
        The F-beta score as a torch tensor.

    Examples
    --------
    >>> conf_mat = np.array([[5, 2], [1, 3]])
    >>> f_beta_from_mat(conf_mat, beta=1.0, num_classes=2, average='micro')
    tensor(0.7273)
    >>> conf_mat = torch.tensor([[5, 2], [1, 3]])
    >>> f_beta_from_mat(conf_mat, beta=1.0, num_classes=2, average='micro')
    tensor(0.7273)
    """

    tp_set = conf_mat.diagonal()
    row = conf_mat.sum(1)
    col = conf_mat.sum(0)

    if average == "micro":
        micro_tp = tp_set.sum()
        micro_fn = (row - tp_set).sum()
        micro_fp = (col - tp_set).sum()

        recall = micro_tp / (micro_tp + micro_fn + EPS)
        precision = micro_tp / (micro_tp + micro_fp + EPS)

        micro_f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + EPS)
        return micro_f1

    elif average == "macro":
        macro_f1 = 0
        for tp, r, c in zip(tp_set, row, col):
            precision = tp / (c + EPS)
            recall = tp / (r + EPS)
            f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + EPS)
            macro_f1 += f1

        return macro_f1 / num_classes
