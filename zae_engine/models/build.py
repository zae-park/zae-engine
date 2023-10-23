import sys
from typing import Optional, Union, Tuple, List, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zae_engine import nn_night as nnn
from zae_engine.models.utility import transformer_option


class CNNBase(nn.Module):
    """
    The interface class using Convolutional Layers.
    """

    def __init__(self):
        super(CNNBase, self).__init__()
        self.conv_api = None

    def gen_block(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: Union[Iterable, int],
        order: int,
        dilation: int = 1,
    ):
        """
        Return block, accumulated layers in-stage. Stack CBR 'order' times.
        :param ch_in: [Int] input channels of block
        :param ch_out: [Int] output channels of block
        :param kernel_size: [Int, Iterable] convolution kernel size.
            If type of argument is int, assume that model will receive 1-D input tensor.
            Else, the type of argument is iterable, assume that model will receive len(argument)-D input tensor.
        :param order: [Int] the number of blocks in stage.
            The stage means same resolution. e.g. from after previous pooling (or stem) to before next pooling.
        :param dilation: [Int, Iterable] Default is 1. If argument more than 1, dilated convolution will performed.
        :return: [nn.Module]
        """

        if isinstance(kernel_size, int):
            self.conv_api = nn.Conv1d
        elif isinstance(kernel_size, Iterable):
            dim = len(kernel_size)
            self.conv_api = nn.Conv2d if dim == 2 else nn.Conv3d if dim == 3 else None
        else:
            raise IndexError("Unexpected shape error.")

        blk = []  # List of blocks
        for o in range(order):  # stack blocks 'order' times
            sequence = nnn.Residual(
                self.unit_layer(ch_in, ch_out, kernel_size, dilation=dilation),
                self.unit_layer(ch_out, ch_out, kernel_size, dilation=dilation),
                nnn.SE(ch_out, reduction=8),
            )
            blk.append(sequence)
        return nn.Sequential(*blk)

    def gen_head(self, c_in, kernel=None):
        if kernel is None:
            kernel = self.kernel_size
        return self.conv_api(
            c_in, self.ch_out, kernel_size=kernel, padding=(kernel - 1) // 2
        )

    def unit_layer(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: Union[Iterable, int],
        stride: int = 1,
        dilation: int = 1,
    ):
        """
        Return unit layer. The unit layer consists of {convolution} - {batch norm} - {activation}.
        :param ch_in: [Int]
        :param ch_out: [Int]
        :param kernel_size: [Int, Iterable]
        :param stride: [Int]
        :param dilation: [Int]
        :return:
        """
        conv1 = self.conv_api(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            bias=True,
        )
        return nn.Sequential(*[conv1, nn.BatchNorm1d(ch_out), nn.ReLU()])


class Segmentor1D(CNNBase):
    """
    Builder class for beat_segmentation which has U-Net-like structure.
    :param ch_in: int
        The channels-wise dimension of input tensor.
    :param ch_out: int
        The channels-wise dimension of output tensor, same as number of classes.
    :param width: int
        The channel expand factor.
    :param kernel_size: int
    :param depth: int
        The number of pooling layers. Must be 1 less than length of stride.
    :param order: int
        The number of blocks in stage. The stage means layer sequence in a depth.
    :param stride: list or tuple
        The scaling ratio for pooling layers.
    :param decoding: bool
        Optional.
        If True, build the decoding part of U-Net.
        If not, replace the decoding part of U-Net to up-sampling.
    :param expanding: bool, optional
        Optional.
        If True, input tensor padded 30 samples bi-side along the spatial axis to match the shape and product of stride.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        width: int,
        kernel_size: int or tuple,
        depth: int,
        order: int,
        stride: list or tuple,
        decoding: bool,
        **kwargs,
    ):
        super(Segmentor1D, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.width = width
        self.kernel_size = kernel_size
        self.depth = depth
        self.order = order
        self.stride = stride
        self.decoding = decoding
        self.kwargs = kwargs

        self.expanding = kwargs["expanding"] if "expanding" in kwargs.keys() else False

        # Encoder (down-stream)
        self.pools = nn.ModuleList()
        self.enc = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.enc.append(
                    nn.Sequential(
                        self.unit_layer(ch_in, width, kernel_size),
                        self.gen_block(width, width, kernel_size, order),
                    )
                )
            else:
                if d == 1:
                    c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
                else:
                    c_in, c_out, s = ch_in + width * d, width * (d + 1), stride[d - 1]
                self.pools.append(
                    nn.AvgPool1d(np.prod(stride[:d]), np.prod(stride[:d]))
                )
                self.enc.append(
                    nn.Sequential(
                        self.unit_layer(c_in, c_out, kernel_size, stride=s),
                        self.gen_block(c_out, c_out, kernel_size, order),
                    )
                )

        self.un_pools = nn.ModuleList()
        for s in stride[::-1]:
            self.un_pools.append(
                nn.Upsample(scale_factor=s, mode="linear", align_corners=True)
            )

        # Decoder (up-stream)
        if self.decoding:
            self.dec = nn.ModuleList()
            for d in reversed(range(1, depth)):
                c_in, c_out = (2 * d + 1) * width, d * width
                self.dec.append(
                    nn.Sequential(
                        self.unit_layer(c_in, c_out, kernel_size),
                        self.gen_block(c_out, c_out, kernel_size, order),
                    )
                )

        self.head = self.gen_head(
            width if decoding else width * depth, kernel=kernel_size
        )

    def forward(self, x):
        if self.expanding:
            x = F.pad(x, (30, 30), mode="constant", value=0)

        for_skip, for_feat = [], []

        # -------------Encoder#------------- #
        for d in range(self.depth):  # 0, 1, 2, 3
            if d == 0:
                for_skip.append(self.enc[d](x))
            elif d == 1:
                for_skip.append(self.enc[d](for_skip[-1]))
            else:
                for_skip.append(
                    self.enc[d](torch.cat([for_skip[-1], self.pools[d - 2](x)], 1))
                )

        # -------------Decoder#------------- #
        if not self.decoding:
            out = self.head(for_skip[-1])
            for un_pool in self.un_pools:
                out = un_pool(out)
        else:
            for_feat.append(for_skip.pop(-1))
            for_skip.reverse()
            for d in range(self.depth - 1):  # 0, 1, 2
                concat = torch.cat((for_skip[d], self.un_pools[d](for_feat[d])), dim=1)
                for_feat.append(self.dec[d](concat))
            out = self.head(for_feat[-1])
        if self.expanding:
            out = (
                tuple([o[:, :, 30:-30] for o in out])
                if isinstance(out, tuple)
                else out[:, :, 30:-30]
            )
        return out


class Regressor1D(CNNBase):
    """
    Builder class for rpeak_regression which has cascade CNN structure.
    :param dim_in: int
        The spatial-wise dimension of input tensor.
    :param ch_in: int
        The channels-wise dimension of input tensor.
    :param width: int
        The channel expand factor.
    :param kernel_size: int
    :param order: int
        The number of blocks in stage. The stage means layer sequence in a depth.
    :param depth: int
        The number of pooling layers. Must be 1 less than length of stride.
    :param stride: list or tuple
        The scaling ratio for pooling layers.
    :param head_depth: int
        The number of layers in head.
    :param embedding_dims: int
        The spatial-wise dimension of feature vector (or latent space).

    """

    def __init__(
        self,
        dim_in: int,
        ch_in: int,
        width: int,
        kernel_size: int or tuple,
        order: int,
        depth: int,
        stride: int,
        head_depth: int,
        embedding_dims: int,
        **kwargs,
    ):
        super(RPeakRegress, self).__init__()

        self.width = width
        self.kernel_size = kernel_size
        self.order = order
        self.depth = depth
        self.stride = stride
        self.head_depth = head_depth
        self.dim = dim_in
        self.embedding_dims = embedding_dims
        self.kwargs = kwargs

        # Encoder (down-stream)
        enc = []
        for d in range(depth):
            c_in = ch_in if d == 0 else width * d
            enc.append(
                self.unit_layer(c_in, width * (d + 1), self.kernel_size, self.stride)
            )
            enc.append(
                self.gen_block(
                    width * (d + 1), width * (d + 1), kernel_size, order=order
                )
            )
        self.enc = nn.Sequential(*enc)

        self.update_dimension()
        self.emb_layer = self.gen_embedding()
        self.head = nn.Linear(self.embedding_dims, 1)
        self.clipper = nnn.ClippedReLU(1)

    def update_dimension(self):
        for _ in range(self.depth):
            self.dim = (
                self.dim + (self.kernel_size - 1) // 2 * 2 - (self.kernel_size - 1) - 1
            ) // self.stride + 1
            # self.dim = 1 + (self.dim + 2*(self.kernel_size//2) - self.kernel_size) // self.stride

    def gen_embedding(self):
        block = []
        for h in range(self.head_depth):
            d_in = (
                self.dim * self.width * self.depth
                if h == 0
                else int(self.dim // 2 ** (self.head_depth - h))
            )
            block.append(nn.Linear(d_in, self.embedding_dims))
            block.append(nn.ReLU())
        return nn.Sequential(*block)

    def unit_layer(self, ch_in, ch_out, kernel_size, stride=1, dilation=1):
        conv1 = nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=(stride,),
            dilation=(dilation,),
            padding=dilation * (kernel_size - 1) // 2,
            bias=True,
        )
        return nn.Sequential(*[conv1, nn.ReLU()])

    def forward(self, x):
        # -------------Encoder#------------- #
        x = self.enc(x)
        flat = x.view(x.shape[0], -1)
        emb = self.emb_layer(flat)
        return self.head(emb)


class Classifier1D(nn.Module):
    """
    Builder class for 10 sec classification model which has Residual Network structure.
    :param num_layers: int
        The number of entire layers. Support the value in [10, 14, 18, 34, 50, 101, 152].
    :param num_classes: int
        The number of classes. Same as a channels-wise dimension of the output tensor.
    :param num_channels: int
        The channels-wise dimension of the input tensor.
    :param kernel_size: int
    :param dropout_rate: int
    :param width_factor: int
        The channel expand factor.
    :param stride: list or tuple
        The scaling ratio for pooling layers.
    :param reduction: int
        The Squeeze and Excitation ratio. Note that it must be a divisor of appropriate num_channels.
    :param bias: bool
        (사용되지 않지만 일단 설명 추가함.)
        If True, layers in model have extra weights (bias).
    :param contrastive: bool
        Only for training.
        During training process, use extra head for representation learning.
    :param first_block: bool, optional
        Optional.
        If True, the input tensor is normalized before passing into skip-connection in the first block of every stage.
        Otherwise, the input tensor is passed as-is to skip-connection.
    :param use_transformer: bool, optional
        Optional.
        If True, the feature map is recalibrated using a transformer manner.
    """

    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        num_channels: int,
        kernel_size: int,
        dropout_rate: float,
        width_factor: int,
        stride: int,
        reduction: int,
        bias: bool = False,
        contrastive: Optional[bool] = False,
        **kwargs,
    ):
        super(Classifier1D, self).__init__()
        num_layers = num_layers
        num_classes = num_classes
        num_channels = num_channels
        self.kernel_size = kernel_size
        k = width_factor
        nGroups = [16 * k, 16 * k, 32 * k, 64 * k, 128 * k]
        self.tmp_ch = nGroups[0]
        self.reduction = reduction
        self.bias = bias
        self.stride = stride

        if "first_block" in kwargs.keys():
            self.first_block = kwargs["first_block"]
        if "use_transformer" in kwargs.keys():
            self.use_transformer = kwargs["use_transformer"]

        supported_num_layers = [10, 14, 18, 34, 50, 101, 152]
        if num_layers not in supported_num_layers:
            num_layers = supported_num_layers[0]
        self.block_type = "basic" if num_layers < 50 else "bottleneck"

        if num_layers == 10:
            num_block_list = [1, 1, 1, 1]
        elif num_layers == 14:
            num_block_list = [2, 1, 1, 2]
        elif num_layers == 18:
            num_block_list = [2, 2, 2, 2]
        elif num_layers == 34:
            num_block_list = [3, 4, 6, 3]
        elif num_layers == 50:
            num_block_list = [3, 4, 6, 3]
        elif num_layers == 101:
            num_block_list = [3, 4, 23, 3]
        elif num_layers == 152:
            num_block_list = [3, 8, 36, 3]
        else:
            num_block_list = []
        self.num_block_list = num_block_list

        self.stem = self.make_stem(
            num_channels, nGroups[0], 7, stride=2, bias=self.bias
        )

        stages = [
            self.stack_layers(
                channel_in=nGroups[i],
                channel_out=nGroups[i + 1],
                dropout_rate=dropout_rate,
                kernel_size=kernel_size,
                num_blocks=num_block_list[i],
                stride=1 if i == 0 else stride,
                first_block=self.first_block if i == 0 else False,
                bias=self.bias,
                reduction=self.reduction,
            )
            for i in range(4)
        ]

        self.stage_1, self.stage_2, self.stage_3, self.stage_4 = stages

        last_channel = self.stage_4[-1].module_list[-2].out_channels

        self.bn1 = nn.BatchNorm1d(last_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if self.use_transformer:
            self.linear = self.get_vit_head(last_channel, num_classes)
        else:
            self.linear = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(last_channel, num_classes),
            )
        if contrastive:
            # projection MLP
            self.projection = self.make_mlp(
                last_channel, last_channel, last_channel, mode="projection"
            )
            # prediction MLP
            self.prediction = self.make_mlp(
                last_channel, last_channel // 4, last_channel, mode="prediction"
            )

    @staticmethod
    def make_mlp(
        in_dim: int, latent_dim: int, out_dim: Optional[int] = None, mode="projection"
    ):
        linear1 = nn.Linear(in_dim, latent_dim)
        bn1 = nn.BatchNorm1d(latent_dim)
        act = nn.ReLU(inplace=True)

        if mode == "projection":
            linear2 = nn.Linear(in_dim, latent_dim)
            bn2 = nn.BatchNorm1d(latent_dim)

            linear3 = nn.Linear(in_dim, latent_dim)
            bn3 = nn.BatchNorm1d(latent_dim)
            return nn.Sequential(linear1, bn1, act, linear2, bn2, act, linear3, bn3)
        elif mode == "prediction":
            linear2 = nn.Linear(latent_dim, out_dim)
            return nn.Sequential(linear1, bn1, act, linear2)

    @staticmethod
    def make_stem(
        channel_in: int, channel_out: int, kernel_size: int, stride: int, bias: bool
    ):
        return nn.Sequential(
            nn.Conv1d(channel_in, channel_out, (kernel_size,), (stride,), bias=bias),
            nn.MaxPool1d(3, padding=1, stride=stride),
        )

    def stack_layers(
        self,
        channel_in: int,
        channel_out: int,
        dropout_rate: float,
        kernel_size: int,
        stride: int,
        num_blocks: int,
        first_block=False,
        bias: bool = False,
        reduction: int = 4,
    ):
        class ResnetBasicBlock(nn.Module):
            """
            ResNet Basic Block
            Supports ResNeXt if cardinality > 1
            -- BN-ReLU-Conv_kx1 - BN-ReLU-Conv_kx1
            -- (GlobalAvgPool - Conv_1x1-ReLU - Conv_1x1-Sigmoid)
            -- MaxPool-Conv_1x1
            """

            def __init__(
                self,
                fe_block: nn.ModuleList,
                shortcut: nn.Sequential,
                first_block: bool,
            ):
                super(ResnetBasicBlock, self).__init__()
                self.module_list = fe_block
                self.first_block = first_block
                self.shortcut = shortcut

            def forward(self, x):
                for i, module in enumerate(self.module_list):
                    if self.first_block and i <= 2:
                        x = module(x)
                    if i == 0:
                        out = module(x)
                    else:
                        out = module(out)

                x = self.shortcut(x)

                out_l, x_l = out.shape[2], x.shape[2]
                if out_l != x_l:
                    x = F.pad(x, (0, out_l - x_l))

                out_c, x_c = out.shape[1], x.shape[1]
                if out_c == x_c:
                    out += x
                else:
                    out += F.pad(x, (0, 0, 0, out_c - x_c))

                return out

        layers = []

        for b in range(num_blocks):
            if b == 0:
                fe, shortcut = self.make_fe_se_block(
                    channel_in=channel_in,
                    channel_out=channel_out,
                    stride=stride,
                    kernel_size=kernel_size,
                    bias=bias,
                    dropout_rate=dropout_rate,
                    reduction=reduction,
                )
                layers.append(ResnetBasicBlock(fe, shortcut, first_block=first_block))
            else:
                fe, shortcut = self.make_fe_se_block(
                    channel_in=channel_out,
                    channel_out=channel_out,
                    stride=1,
                    kernel_size=kernel_size,
                    bias=bias,
                    dropout_rate=dropout_rate,
                    reduction=reduction,
                )
                layers.append(ResnetBasicBlock(fe, shortcut, first_block=False))

        return nn.Sequential(*layers)

    def make_fe_se_block(
        self,
        channel_in: int,
        channel_out: int,
        stride: int,
        kernel_size: int,
        bias: bool,
        dropout_rate: float,
        reduction: int,
    ):
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)

        fe_block = nn.ModuleList(
            [
                nn.BatchNorm1d(channel_in),
                relu,
                nn.Conv1d(
                    channel_in,
                    channel_out,
                    kernel_size=(kernel_size,),
                    stride=stride,
                    bias=bias,
                )
                if stride != 1
                else nn.Conv1d(
                    channel_in, channel_out, (kernel_size,), padding="same", bias=bias
                ),
                nn.BatchNorm1d(channel_out),
                relu,
                nn.Dropout(dropout_rate),
                nn.Conv1d(
                    channel_out, channel_out, (kernel_size,), padding="same", bias=bias
                ),
                nnn.SE(
                    channel_out, reduction=reduction, bias=bias
                ),  # TODO: make as an option
            ]
        )

        shortcut = nn.Sequential(
            nn.MaxPool1d(stride) if stride != 1 else nn.Identity(),
            nn.Conv1d(channel_in, channel_out, kernel_size=(1,), bias=bias)
            if channel_in != channel_out and self.block_type == "conv"
            else nn.Identity(),
        )

        return fe_block, shortcut

    @staticmethod
    def get_vit_head(in_ch, num_class):
        # image_size, patch_size, num_classes, dim, depth, heads, mlp_dim
        linear = nnn.TransformerLayer(
            image_size=66,
            patch_size=1,
            num_classes=num_class,
            depth=3,
            heads=4,
            mlp_dim=256,
            tmp_ch=in_ch,
        )
        linear.apply(transformer_option)

        return linear

    def forward(self, x, model_type="classifier"):
        out = self.stem(x)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)

        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        if model_type == "classifier":
            out = self.linear(out)
            return out
        else:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            # projection
            z = self.projection(out)
            # prediction
            p = self.prediction(out)
            return z, p
