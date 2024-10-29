from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from . import resblock, conv_block


class UNetBlock(resblock.BasicBlock):
    """
    Two times of [Conv-normalization-activation] block for UNet architecture.

    This module is a modified version of the BasicBlock used in ResNet, adapted for the UNet architecture.

    Parameters
    ----------
    ch_in : int
        The number of input channels.
    ch_out : int
        The number of output channels.
    stride : int, optional
        The stride of the convolution. Default is 1.
    groups : int, optional
        The number of groups for the convolution. Default is 1.
    norm_layer : Callable[..., nn.Module], optional
        The normalization layer to use. Default is `nn.BatchNorm2d`.

    Attributes
    ----------
    expansion : int
        The expansion factor of the block, set to 1.

    References
    ----------
    .. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
           In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR) (pp. 770-778).
           https://arxiv.org/abs/1512.03385
    .. [2] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
           In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241).
           https://arxiv.org/abs/1505.04597
    """

    expansion: int = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(ch_in=ch_in, ch_out=ch_out, stride=1, groups=groups, dilation=1, norm_layer=norm_layer)
        self.downsample = None if stride == 1 else nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the UNetBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after applying the block operations.
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = out if self.downsample is None else self.downsample(out)
        return out


class RSU7(nn.Module):
    expansion = 2
    """
    U²-Net's RSU7 (Recurrent Residual U-block with 7 levels).
    """

    def __init__(self, ch_in: int, ch_out: int, width: int = None, *args, **kwargs):
        super(RSU7, self).__init__()
        if width is None:
            width = ch_out // self.expansion
        self.rebnconvin = conv_block.ConvBlock(ch_in=ch_in, ch_out=ch_out, dilate=1)

        self.rebnconv1 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)
        self.pool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv7 = conv_block.ConvBlock(ch_in=ch_out, ch_out=width)

        self.rebnconv6d = conv_block.ConvBlock(ch_in=width * 2, ch_out=width)
        self.rebnconv5d = conv_block.ConvBlock(ch_in=width * 2, ch_out=width)
        self.rebnconv4d = conv_block.ConvBlock(ch_in=width * 2, ch_out=width)
        self.rebnconv3d = conv_block.ConvBlock(ch_in=width * 2, ch_out=width)
        self.rebnconv2d = conv_block.ConvBlock(ch_in=width * 2, ch_out=width)
        self.rebnconv1d = conv_block.ConvBlock(ch_in=width * 2, ch_out=ch_out)

        self.rebnconvout = conv_block.ConvBlock(width + ch_out, ch_out, dilate=1)

    def forward(self, x):
        """
        Forward pass for RSU7.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying RSU7 operations.
        """
        hx = x
        hxin = self.rebnconvin(hx)

        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)

        h2 = self.rebnconv2(h)
        h = self.pool2(h2)

        h3 = self.rebnconv3(h)
        h = self.pool3(h3)

        h4 = self.rebnconv4(h)
        h = self.pool4(h4)

        h5 = self.rebnconv5(h)
        h = self.pool5(h5)

        h6 = self.rebnconv6(h)
        h = self.pool6(h6)

        h7 = self.rebnconv7(h)

        # Decoder path
        h6d = self.rebnconv6d(torch.cat((h7, h6), 1))
        h6d = F.interpolate(h6d, scale_factor=2, mode="bilinear", align_corners=True)

        h5d = self.rebnconv5d(torch.cat((h6d, h5), 1))
        h5d = F.interpolate(h5d, scale_factor=2, mode="bilinear", align_corners=True)

        h4d = self.rebnconv4d(torch.cat((h5d, h4), 1))
        h4d = F.interpolate(h4d, scale_factor=2, mode="bilinear", align_corners=True)

        h3d = self.rebnconv3d(torch.cat((h4d, h3), 1))
        h3d = F.interpolate(h3d, scale_factor=2, mode="bilinear", align_corners=True)

        h2d = self.rebnconv2d(torch.cat((h3d, h2), 1))
        h2d = F.interpolate(h2d, scale_factor=2, mode="bilinear", align_corners=True)

        h1d = self.rebnconv1d(torch.cat((h2d, h1), 1))
        h1d = F.interpolate(h1d, scale_factor=2, mode="bilinear", align_corners=True)

        return self.rebnconvout(torch.cat((h1d, hxin), 1))
