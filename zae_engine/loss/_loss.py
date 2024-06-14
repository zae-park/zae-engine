from typing import Union, Optional

import torch
import torch.nn.functional as F


def cross_entropy(logit: torch.Tensor, y_hot: torch.Tensor, class_weights: Union[torch.Tensor, None] = None):
    """
    Compute the binary cross-entropy loss with logits.

    Parameters
    ----------
    logit : torch.Tensor
        The predicted logits.
    y_hot : torch.Tensor
        The one-hot encoded true labels.
    class_weights : Union[torch.Tensor, None], optional
        A manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`.

    Returns
    -------
    torch.Tensor
        The computed binary cross-entropy loss.
    """
    loss = F.binary_cross_entropy_with_logits(logit, y_hot.float(), weight=class_weights)
    return loss


def batch_wise_dot(batch: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Compute the dot-product for all combinations of vectors in a batch.

    Parameters
    ----------
    batch : torch.Tensor
        The input batch tensor with shape [Batch, N] where N is the length of each vector.
    reduce : bool, optional
        If True, returns a matrix with dot-product values for each vector combination in the batch.
        If False, returns the average of the dot-product values.

    Returns
    -------
    torch.Tensor
        If reduce is True, returns a tensor of shape [Batch, Batch] where each value represents
        the dot-product between two vectors.
        If reduce is False, returns a tensor representing the average of the dot-product values.
    """
    norm = torch.norm(batch, dim=1, keepdim=True)
    normalized = (batch / norm).unsqueeze(1)  # [batch, 1, N]

    mat1 = normalized.permute(2, 0, 1)  # [N, batch, 1]
    mat2 = normalized.permute(2, 1, 0)  # [N, 1, batch]
    squared_mat = torch.bmm(mat1, mat2)  # [N, batch, batch]. 1st dimension represents element-wise product vector

    dot_mat = squared_mat.sum(None if reduce else 0)
    return dot_mat
