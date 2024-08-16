from typing import Optional

import torch.nn as nn

from ..builds import transformer as trx
from ...loss import angular

#
# from ..builds.legacy import Segmentor, Regressor1D
# from ..utility import initializer, WeightLoader
#
#
# def beat_segmentation(pretrained: Optional[bool] = False) -> nn.Module:
#     """
#     Build model which has a same structure with the latest released model.
#     :param pretrained: bool
#         If True, load weight from the server.
#         If not, weights are initialized with 'initializer' method in utils.py
#     :return: nn.Module
#     """
#     model = Segmentor(
#         ch_in=1,
#         ch_out=4,
#         width=16,
#         kernel_size=7,
#         depth=5,
#         order=4,
#         stride=(2, 2, 2, 2),
#         decoding=False,
#     )
#
#     if pretrained:
#         weights = WeightLoader.get("beat_segmentation")
#         model.load_state_dict(weights, strict=True)
#     else:
#         model.apply(initializer)
#
#     return model
#
#
# def peak_regression(pretrained: Optional[bool] = False) -> nn.Module:
#     """
#     Build model which has a same structure with the latest released model.
#     :param pretrained: bool
#         If True, load weight from the server.
#         If not, weights are initialized with 'initializer' method in utils.py.
#     :return: nn.Module
#     """
#     model = Regressor1D(
#         dim_in=64,
#         ch_in=1,
#         width=16,
#         kernel_size=3,
#         depth=2,
#         stride=1,
#         order=4,
#         head_depth=1,
#         embedding_dims=16,
#     )
#
#     if pretrained:
#         weights = WeightLoader.get("peak_regression")
#         model.load_state_dict(weights, strict=True)
#     else:
#         model.apply(initializer)
#
#     return model
#
#
# def u_net(pretrained: Optional[bool] = False) -> nn.Module:
#     model = Segmentor(
#         ch_in=1,
#         ch_out=2,
#         width=64,
#         kernel_size=[3, 3],
#         depth=5,
#         order=2,
#         stride=(2, 2, 2, 2),
#         decoding=True,
#     )
#     if pretrained:
#         weights = WeightLoader.get("u_net")
#         model.load_state_dict(weights, strict=True)
#
#     return model
#
#
# # if __name__ == "__main__":
# #     import numpy as np
# #
# #     m = u_net()
# #     tmp = np.zeros((10, 1, 256, 256), dtype=np.float32)
# #     tmp = torch.Tensor(tmp)
# #     p = m(tmp)
# #     print(m)
# #     print(p)
# #
# #     m = u_net1()
# #     tmp = np.zeros((10, 1, 256), dtype=np.float32)
# #     tmp = torch.Tensor(tmp)
# #     p = m(tmp)
# #     print(m)
# #     print(p)


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
        self.transformer = trx.TimeAwareTransformer(
            d_head=d_head, d_model=d_model, n_head=n_head, num_layers=num_layers
        )
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
