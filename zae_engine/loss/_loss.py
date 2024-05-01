from typing import Union

import torch
import torch.nn.functional as F


def cross_entropy(logit: torch.Tensor, y_hot: torch.Tensor, class_weights: Union[torch.Tensor, None] = None):
    loss = F.binary_cross_entropy_with_logits(logit, y_hot.float(), weight=class_weights)
    return loss


def batch_wise_dot(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute dot-product for all combination of vectors in batch.
    Possible shape of batch is [Batch, N] where N is length of dimension.
    :param batch:
    :return:
    """
    norm = torch.norm(batch, dim=1, keepdim=True)
    normalized = (batch / norm).unsqueeze(1)  # [batch, 1, N]

    mat1 = normalized.permute(2, 0, 1)  # [N, batch, 1]
    mat2 = normalized.permute(2, 1, 0)  # [N, 1, batch]
    squared_mat = torch.bmm(mat1, mat2)  # [N, batch, batch]. 1st dimension represents element-wise product vector
    dot_mat = squared_mat.sum(0)  # [batch, batch]. Each values represent dot-product value between two vector.
    return torch.mean(dot_mat)  # - torch.mean(eye)
