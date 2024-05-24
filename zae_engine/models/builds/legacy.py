# from typing import Union, Tuple, Iterable
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from ... import nn_night as nnn
#
#
# class CNNBaseLegacy(nn.Module):
#     """
#     The interface class using Convolutional Layers.
#     """
#
#     def __init__(
#         self,
#         ch_in: int,
#         ch_out: int,
#         width: int,
#         kernel_size: int or tuple,
#         depth: int,
#         order: int,
#         stride: list or tuple,
#     ):
#         super(CNNBaseLegacy, self).__init__()
#         self.ch_in = ch_in
#         self.ch_out = ch_out
#         self.width = width
#         self.kernel_size = kernel_size
#         self.depth = depth
#         self.order = order
#         self.stride = stride
#
#         self.set_dimension()
#         self.body, self.pools = self.gen_body(
#             ch_in=self.ch_in,
#             width=self.width,
#             kernel_size=self.kernel_size,
#             depth=self.depth,
#             order=self.order,
#             stride=self.stride,
#         )
#
#     def set_dimension(self):
#         if isinstance(self.kernel_size, int):
#             self._dim = 1
#             self._conv = nn.Conv1d
#             self._bn = nn.BatchNorm1d
#             self._pool = nn.AvgPool1d
#         else:
#             self._dim = len(self.kernel_size)
#             if self._dim > 3:
#                 raise IndexError("Unexpected shape error.")
#             self._conv = nn.Conv2d if self._dim == 2 else nn.Conv3d if self._dim == 3 else None
#             self._bn = nn.BatchNorm2d if self._dim == 2 else nn.BatchNorm3d if self._dim == 3 else None
#             self._pool = nn.AvgPool2d if self._dim == 2 else nn.AvgPool3d if self._dim == 3 else None
#
#     def calc_padding(self, dilation, kernels):
#         if isinstance(kernels, int):
#             return dilation * (kernels - 1) // 2
#         else:
#             return [dilation * (k - 1) // 2 for k in kernels]
#
#     def unit_layer(
#         self,
#         ch_in: int,
#         ch_out: int,
#         kernel_size: Union[Iterable, int],
#         stride: int = 1,
#         dilation: int = 1,
#     ):
#         """
#         Return unit layer. The unit layer consists of {convolution} - {batch norm} - {activation}.
#         :param ch_in: [Int]
#         :param ch_out: [Int]
#         :param kernel_size: [Int, Iterable]
#         :param stride: [Int]
#         :param dilation: [Int]
#         :return:
#         """
#         conv1 = self._conv(
#             ch_in,
#             ch_out,
#             kernel_size=kernel_size,
#             stride=stride,
#             dilation=dilation,
#             padding=self.calc_padding(dilation, kernel_size),
#             bias=True,
#         )
#         return nn.Sequential(*[conv1, self._bn(ch_out), nn.ReLU()])
#
#     def gen_block(
#         self,
#         ch_in: int,
#         ch_out: int,
#         kernel_size: Union[int, list, tuple],
#         order: int,
#         dilation: int = 1,
#     ):
#         """
#         Return block, accumulated layers in-stage. Stack CBR 'order' times.
#         :param ch_in: [Int] input channels of block
#         :param ch_out: [Int] output channels of block
#         :param kernel_size: [Int, Iterable] convolution kernel size.
#             If type of argument is int, assume that model will receive 1-D input tensor.
#             Else, the type of argument is iterable, assume that model will receive len(argument)-D input tensor.
#         :param order: [Int] the number of blocks in stage.
#             The stage means same resolution. e.g. from after previous pooling (or stem) to before next pooling.
#         :param dilation: [Int, Iterable] Default is 1. If argument more than 1, dilated convolution will performed.
#         :return: [nn.Module]
#         """
#
#         blk = []  # List of blocks
#         for o in range(order):  # stack blocks 'order' times
#             sequence = nnn.Residual(
#                 self.unit_layer(ch_in, ch_out, kernel_size, dilation=dilation),
#                 self.unit_layer(ch_out, ch_out, kernel_size, dilation=dilation),
#             )
#             blk.append(sequence)
#         return nn.Sequential(*blk)
#
#     def gen_body(self, ch_in, width, kernel_size, depth, order, stride) -> Tuple[nn.Sequential, nn.Sequential]:
#         # Feature extraction
#         pools = nn.ModuleList()
#         body = nn.ModuleList()
#
#         for d in range(depth):
#             if d == 0:
#                 body.append(
#                     nn.Sequential(
#                         self.unit_layer(ch_in, width, kernel_size),
#                         self.gen_block(width, width, kernel_size, order),
#                     )
#                 )
#             else:
#                 if d == 1:
#                     c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
#                 else:
#                     c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
#                 pools.append(self._pool(p := (int(np.prod(stride[:d]))), p))
#                 body.append(
#                     nn.Sequential(
#                         self.unit_layer(c_in, c_out, kernel_size, stride=s),
#                         self.gen_block(c_out, c_out, kernel_size, order),
#                     )
#                 )
#         return nn.Sequential(*body), nn.Sequential(*pools)
#
#     def gen_head(self, ch_in, ch_out, kernel=None):
#         if kernel is None:
#             kernel = self.kernel_size
#         return self._conv(ch_in, ch_out, kernel_size=kernel, padding="same")
#
#
# class Segmentor(CNNBaseLegacy):
#     """
#     Builder class for beat_segmentation which has U-Net-like structure.
#     :param ch_in: int
#         The channels-wise dimension of input tensor.
#     :param ch_out: int
#         The channels-wise dimension of output tensor, same as number of classes.
#     :param width: int
#         The channel expand factor.
#     :param kernel_size: int
#     :param depth: int
#         The number of pooling layers. Must be 1 less than length of stride.
#     :param order: int
#         The number of blocks in stage. The stage means layer sequence in a depth.
#     :param stride: list or tuple
#         The scaling ratio for pooling layers.
#     :param decoding: bool
#         Optional.
#         If True, build the decoding part of U-Net.
#         If not, replace the decoding part of U-Net to up-sampling.
#     """
#
#     def __init__(
#         self,
#         ch_in: int = 1,
#         ch_out: int = 2,
#         width: int = 16,
#         kernel_size: int or tuple = 3,
#         depth: int = 5,
#         order: int = 2,
#         stride: list or tuple = (2, 2, 2, 2),
#         decoding: bool = False,
#     ):
#         super(Segmentor, self).__init__(ch_in, ch_out, width, kernel_size, depth, order, stride)
#         self.enc = self.body
#         self.decoding = decoding
#
#         self.un_pools = nn.ModuleList()
#         for s in stride[::-1]:
#             self.un_pools.append(
#                 nn.Upsample(
#                     scale_factor=tuple([s] * self._dim),
#                     mode="linear" if self._dim == 1 else "bilinear",
#                     align_corners=True,
#                 )
#             )
#
#         # Decoder (up-stream)
#         if decoding:
#             self.dec = nn.ModuleList()
#             for d in reversed(range(1, depth)):
#                 c_in, c_out = (2 * d + 1) * width, d * width
#                 self.dec.append(
#                     nn.Sequential(
#                         self.unit_layer(c_in, c_out, kernel_size),
#                         self.gen_block(c_out, c_out, kernel_size, order),
#                     )
#                 )
#
#         self.head = self.gen_head(width if decoding else width * depth, ch_out, kernel=kernel_size)
#
#     def forward(self, x):
#         for_skip, for_feat = [], []
#
#         # -------------Encoder#------------- #
#         for d in range(self.depth):  # 0, 1, 2, 3
#             if d == 0:
#                 for_skip.append(self.enc[d](x))
#             elif d == 1:
#                 for_skip.append(self.enc[d](for_skip[-1]))
#             else:
#                 for_skip.append(self.enc[d](torch.cat([for_skip[-1], self.pools[d - 2](x)], 1)))
#
#         # -------------Decoder#------------- #
#         if not self.decoding:
#             out = self.head(for_skip[-1])
#             for un_pool in self.un_pools:
#                 out = un_pool(out)
#         else:
#             for_feat.append(for_skip.pop(-1))
#             for_skip.reverse()
#             for d in range(self.depth - 1):  # 0, 1, 2
#                 concat = torch.cat((for_skip[d], self.un_pools[d](for_feat[d])), dim=1)
#                 for_feat.append(self.dec[d](concat))
#             out = self.head(for_feat[-1])
#         return out
#
#
# class Regressor1D(CNNBaseLegacy):
#     """
#     Builder class for rpeak_regression which has cascade CNN structure.
#     :param dim_in: int
#         The spatial-wise dimension of input tensor.
#     :param ch_in: int
#         The channels-wise dimension of input tensor.
#     :param width: int
#         The channel expand factor.
#     :param kernel_size: int
#     :param order: int
#         The number of blocks in stage. The stage means layer sequence in a depth.
#     :param depth: int
#         The number of pooling layers. Must be 1 less than length of stride.
#     :param stride: list or tuple
#         The scaling ratio for pooling layers.
#     :param head_depth: int
#         The number of layers in head.
#     :param embedding_dims: int
#         The spatial-wise dimension of feature vector (or latent space).
#
#     """
#
#     def __init__(
#         self,
#         dim_in: int,
#         ch_in: int,
#         width: int,
#         kernel_size: int or tuple,
#         order: int,
#         depth: int,
#         stride: int,
#         head_depth: int,
#         embedding_dims: int,
#         **kwargs,
#     ):
#         super(Regressor1D, self).__init__()
#
#         self.width = width
#         self.kernel_size = kernel_size
#         self.order = order
#         self.depth = depth
#         self.stride = stride
#         self.head_depth = head_depth
#         self.dim = dim_in
#         self.embedding_dims = embedding_dims
#         self.kwargs = kwargs
#
#         # Encoder (down-stream)
#         enc = []
#         for d in range(depth):
#             c_in = ch_in if d == 0 else width * d
#             enc.append(self.unit_layer(c_in, width * (d + 1), self.kernel_size, self.stride))
#             enc.append(self.gen_block(width * (d + 1), width * (d + 1), kernel_size, order=order))
#         self.enc = nn.Sequential(*enc)
#
#         self.update_dimension()
#         self.emb_layer = self.gen_embedding()
#         self.head = nn.Linear(self.embedding_dims, 1)
#         self.clipper = nnn.ClippedReLU(1)
#
#     def update_dimension(self):
#         for _ in range(self.depth):
#             self.dim = (self.dim + (self.kernel_size - 1) // 2 * 2 - (self.kernel_size - 1) - 1) // self.stride + 1
#             # self.dim = 1 + (self.dim + 2*(self.kernel_size//2) - self.kernel_size) // self.stride
#
#     def gen_embedding(self):
#         block = []
#         for h in range(self.head_depth):
#             d_in = self.dim * self.width * self.depth if h == 0 else int(self.dim // 2 ** (self.head_depth - h))
#             block.append(nn.Linear(d_in, self.embedding_dims))
#             block.append(nn.ReLU())
#         return nn.Sequential(*block)
#
#     def unit_layer(self, ch_in, ch_out, kernel_size, stride=1, dilation=1):
#         conv1 = nn.Conv1d(
#             ch_in,
#             ch_out,
#             kernel_size=kernel_size,
#             stride=(stride,),
#             dilation=(dilation,),
#             padding=dilation * (kernel_size - 1) // 2,
#             bias=True,
#         )
#         return nn.Sequential(*[conv1, nn.ReLU()])
#
#     def forward(self, x):
#         # -------------Encoder#------------- #
#         x = self.enc(x)
#         flat = x.view(x.shape[0], -1)
#         emb = self.emb_layer(flat)
#         return self.head(emb)
#
#
# class ResNet1D(nn.Module):
#     """
#     Builder class for resnet model.
#     :param num_layers: int
#         The number of entire layers. Support the value in [10, 14, 18, 34, 50, 101, 152].
#     :param num_classes: int
#         The number of classes. Same as a channels-wise dimension of the output tensor.
#     :param num_channels: int
#         The channels-wise dimension of the input tensor.
#     :param kernel_size: int
#     :param dropout_rate: int
#     :param width_factor: int
#         The channel expand factor.
#     :param stride: list or tuple
#         The scaling ratio for pooling layers.
#     """
#
#     def __init__(
#         self,
#         num_layers: int,
#         num_classes: int,
#         num_channels: int,
#         kernel_size: int,
#         dropout_rate: float,
#         width_factor: int,
#         stride: int,
#         **kwargs,
#     ):
#         super(ResNet1D, self).__init__()
#         num_layers = num_layers
#         num_classes = num_classes
#         num_channels = num_channels
#         self.kernel_size = kernel_size
#         k = width_factor
#         nGroups = [16 * k, 16 * k, 32 * k, 64 * k, 128 * k]
#         self.tmp_ch = nGroups[0]
#         self.stride = stride
#
#         supported_num_layers = [10, 14, 18, 34, 50, 101, 152]
#         if num_layers not in supported_num_layers:
#             num_layers = supported_num_layers[0]
#         self.block_type = "basic" if num_layers < 50 else "bottleneck"
#
#         if num_layers == 10:
#             num_block_list = [1, 1, 1, 1]
#         elif num_layers == 14:
#             num_block_list = [2, 1, 1, 2]
#         elif num_layers == 18:
#             num_block_list = [2, 2, 2, 2]
#         elif num_layers == 34:
#             num_block_list = [3, 4, 6, 3]
#         elif num_layers == 50:
#             num_block_list = [3, 4, 6, 3]
#         elif num_layers == 101:
#             num_block_list = [3, 4, 23, 3]
#         elif num_layers == 152:
#             num_block_list = [3, 8, 36, 3]
#         else:
#             num_block_list = []
#         self.num_block_list = num_block_list
#
#         self.stem = self.make_stem(num_channels, nGroups[0], 7, stride=2)
#
#         stages = [
#             self.stack_layers(
#                 channel_in=nGroups[i],
#                 channel_out=nGroups[i + 1],
#                 dropout_rate=dropout_rate,
#                 kernel_size=kernel_size,
#                 num_blocks=num_block_list[i],
#                 stride=1 if i == 0 else stride,
#             )
#             for i in range(4)
#         ]
#
#         self.stage_1, self.stage_2, self.stage_3, self.stage_4 = stages
#
#         last_channel = self.stage_4[-1].module_list[-2].out_channels
#
#         self.bn1 = nn.BatchNorm1d(last_channel)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#
#         self.linear = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Linear(last_channel, num_classes),
#         )
#
#     @staticmethod
#     def make_stem(channel_in: int, channel_out: int, kernel_size: int, stride: int):
#         return nn.Sequential(
#             nn.Conv1d(channel_in, channel_out, (kernel_size,), (stride,)),
#             nn.MaxPool1d(3, padding=1, stride=stride),
#         )
#
#     def stack_layers(
#         self,
#         channel_in: int,
#         channel_out: int,
#         dropout_rate: float,
#         kernel_size: int,
#         stride: int,
#         num_blocks: int,
#     ):
#         class ResnetBasicBlock(nn.Module):
#             """
#             ResNet Basic Block
#             Supports ResNeXt if cardinality > 1
#             -- BN-ReLU-Conv_kx1 - BN-ReLU-Conv_kx1
#             -- (GlobalAvgPool - Conv_1x1-ReLU - Conv_1x1-Sigmoid)
#             -- MaxPool-Conv_1x1
#             """
#
#             def __init__(self, fe_block: nn.ModuleList, shortcut: nn.Sequential):
#                 super(ResnetBasicBlock, self).__init__()
#                 self.module_list = fe_block
#                 self.shortcut = shortcut
#
#             def forward(self, x):
#                 for i, module in enumerate(self.module_list):
#                     out = module(x if i == 0 else out)
#                 x = self.shortcut(x)
#
#                 out_l, x_l = out.shape[2], x.shape[2]
#                 if out_l != x_l:
#                     x = F.pad(x, (0, out_l - x_l))
#
#                 out_c, x_c = out.shape[1], x.shape[1]
#                 if out_c == x_c:
#                     out += x
#                 else:
#                     out += F.pad(x, (0, 0, 0, out_c - x_c))
#
#                 return out
#
#         layers = []
#
#         for b in range(num_blocks):
#             fe, shortcut = self.make_fe_block(
#                 channel_in=channel_in if b == 0 else channel_out,
#                 channel_out=channel_out,
#                 stride=stride if b == 0 else 1,
#                 kernel_size=kernel_size,
#                 dropout_rate=dropout_rate,
#             )
#             layers.append(ResnetBasicBlock(fe, shortcut))
#
#         return nn.Sequential(*layers)
#
#     def make_fe_block(
#         self,
#         channel_in: int,
#         channel_out: int,
#         stride: int,
#         kernel_size: int,
#         dropout_rate: float,
#     ):
#         relu = nn.ReLU()
#
#         fe_block = nn.ModuleList(
#             [
#                 nn.BatchNorm1d(channel_in),
#                 relu,
#                 (
#                     nn.Conv1d(
#                         channel_in,
#                         channel_out,
#                         kernel_size=(kernel_size,),
#                         stride=stride,
#                     )
#                     if stride != 1
#                     else nn.Conv1d(channel_in, channel_out, (kernel_size,), padding="same")
#                 ),
#                 nn.BatchNorm1d(channel_out),
#                 relu,
#                 nn.Dropout(dropout_rate),
#                 nn.Conv1d(channel_out, channel_out, (kernel_size,), padding="same"),
#             ]
#         )
#
#         shortcut = nn.Sequential(
#             nn.MaxPool1d(stride) if stride != 1 else nn.Identity(),
#             (
#                 nn.Conv1d(channel_in, channel_out, kernel_size=(1,))
#                 if channel_in != channel_out and self.block_type == "conv"
#                 else nn.Identity()
#             ),
#         )
#
#         return fe_block, shortcut
#
#     def forward(self, x):
#         out = self.stem(x)
#         out = self.stage_1(out)
#         out = self.stage_2(out)
#         out = self.stage_3(out)
#         out = self.stage_4(out)
#
#         out = self.relu(self.bn1(out))
#         out = self.dropout(out)
#
#         out = self.linear(out)
#         return out
