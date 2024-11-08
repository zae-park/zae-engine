from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.functional as F
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


class RSUBlock(nn.Module):
    """
    Recurrent Residual U-block (RSU) implementation.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_mid : int
        Number of middle channels.
    ch_out : int
        Number of output channels.
    height : int
        Number of layers in the RSU block (e.g., RSU4 has height=4, RSU7 has height=7).
    dilation_height : int
        Dilation rate for convolutions within the block.
    pool_size : int, optional
        Pooling kernel size. Default is 2.

    References
    ----------
    .. [1] Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020).
            U2-Net: Going deeper with nested U-structure for salient object detection. Pattern recognition, 106, 107404.
            (https://arxiv.org/pdf/2005.09007)

    """

    def __init__(
        self, ch_in: int, ch_mid: int, ch_out: int, height: int = 7, dilation_height: int = 7, pool_size: int = 2
    ):
        super(RSUBlock, self).__init__()
        assert height >= dilation_height, "dilation_height must be less or equal than height."

        self.height, self.dilation_height = height, dilation_height
        self.minimum_resolution = 2 ** (height - 2)
        self.pool_size = pool_size
        self.stem = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1)

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(1, height):
            is_first = i == 1
            ch_ = ch_out if is_first else ch_mid
            if i >= dilation_height:
                # Dilation mode: use dilation 2 instead of Pooling layer
                self.encoder_blocks.append(conv_block.ConvBlock(ch_in=ch_, ch_out=ch_mid, dilate=2))
                self.pools.append(nn.Identity())
            else:
                # Vanilla mode: use Pooling layer with kernel & stride 2 instead of dilation in Convolutional layer.
                self.encoder_blocks.append(conv_block.ConvBlock(ch_in=ch_, ch_out=ch_mid))
                self.pools.append(self.down_layer(at_first=is_first))

        # Bottleneck block with dilation 2.
        self.bottleneck = conv_block.ConvBlock(ch_in=ch_mid, ch_out=ch_mid, dilate=2)

        # Decoder path
        self.ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in reversed(range(1, height)):
            is_last = i == 1
            ch_ = ch_out if is_last else ch_mid
            if i >= dilation_height:
                # Dilation mode: use dilation 2 instead of Up-sampling layer
                self.ups.append(nn.Identity())
                self.decoder_blocks.append(conv_block.ConvBlock(ch_in=ch_mid * 2, ch_out=ch_, dilate=2))
            else:
                # Vanilla mode: use Up-sampling layer with kernel & stride 2 instead of dilation in Convolutional layer.
                self.ups.append(self.up_layer(at_last=is_last))
                self.decoder_blocks.append(conv_block.ConvBlock(ch_in=ch_mid * 2, ch_out=ch_))

    def down_layer(self, at_first: bool = False):
        """Returns a downsampling layer or identity based on the position."""
        return nn.Identity() if at_first else nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)

    def up_layer(self, at_last: bool = False):
        """Returns an upsampling layer or identity based on the position."""
        return nn.Identity() if at_last else nn.Upsample(scale_factor=self.pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RSUBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_ch, height, width).
        """
        feat = self.stem(x)
        features = [feat]

        # Encoder path
        for enc, down in zip(self.encoder_blocks, self.pools):
            feat = enc(feat)
            features.append(feat)
            feat = down(feat)

        # Bottleneck processing
        feat = self.bottleneck(feat)

        # Decoder path
        for dec, up in zip(self.decoder_blocks, self.ups):
            feat = torch.cat([up(feat), features.pop()], dim=1)
            feat = dec(feat)

        return feat + features.pop()
