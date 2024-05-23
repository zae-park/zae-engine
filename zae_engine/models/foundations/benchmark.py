# from typing import Optional
#
# import torch.nn as nn
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
