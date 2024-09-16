import torch.nn as nn

from ..builds import transformer as trx
from ...loss import angular


class UserIdModel(nn.Module):
    """
    User identification model using a Transformer encoder and ArcFace loss.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    num_classes : int
        Number of user classes.

    Methods
    -------
    forward(event_vecs, time_vecs, labels, mask=None):
        Forward pass through the model.
    expand_classes(new_num_classes):
        Expand the number of classes dynamically.
    """

    def __init__(self, d_head, d_model, n_head, num_layers, num_classes):
        super(UserIdModel, self).__init__()
        self.transformer = trx.EncoderBase(d_model=d_model, num_layers=num_layers)
        self.arcface = angular.ArcFaceLoss(d_model, num_classes)
        self.num_classes = num_classes

    def forward(self, event_vecs, time_vecs, labels, mask=None):
        """
        Forward pass through the UserIdentificationModel.

        Parameters
        ----------
        event_vecs : torch.Tensor
            Input event vectors of shape (seq_len, batch_size, 128).
        time_vecs : torch.Tensor
            Input time vectors of shape (seq_len, batch_size).
        labels : torch.Tensor
            Ground truth labels of shape (batch_size).
        mask : torch.Tensor, optional
            Mask for the input data (default is None).

        Returns
        -------
        tuple of torch.Tensor
            Logits of shape (batch_size, num_classes) and features of shape (batch_size, d_model).
        """
        features = self.transformer(event_vecs, time_vecs, mask)
        logits = self.arcface(features, labels)
        return logits, features

    def expand_classes(self, new_num_classes):
        """
        Expand the number of classes dynamically.

        Parameters
        ----------
        new_num_classes : int
            New number of classes.
        """
        old_weight = self.arcface.weight.data
        self.arcface = angular.ArcFaceLoss(self.arcface.weight.shape[1], new_num_classes)
        self.arcface.weight.data[: self.num_classes] = old_weight
        self.num_classes = new_num_classes
