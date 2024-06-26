import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for classification with angular margin.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Number of classes.
    s : float, optional
        Norm of input feature (default is 30.0).
    m : float, optional
        Margin (default is 0.50).

    Methods
    -------
    forward(features, labels):
        Forward pass to compute the ArcFace loss.

    References
    ----------
    .. [1] Deng, Jiankang, et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
           https://arxiv.org/abs/1801.07698
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super(ArcFaceLoss, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass to compute the ArcFace loss.

        Parameters
        ----------
        features : torch.Tensor
            Input features of shape (batch_size, in_features).
        labels : torch.Tensor
            Ground truth labels of shape (batch_size).

        Returns
        -------
        torch.Tensor
            Computed ArcFace logits of shape (batch_size, out_features).
        """
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
