from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from . import resblock


class UNetBlock(resblock.BasicBlock):
    """
    Residual block for UNet architecture.

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
