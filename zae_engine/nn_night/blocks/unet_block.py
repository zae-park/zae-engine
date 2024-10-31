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
    Recurrent Residual U-block (RSU) 구현.

    Parameters
    ----------
    in_ch : int
        입력 채널 수.
    mid_ch : int
        중간 채널 수.
    out_ch : int
        출력 채널 수.
    height : int
        RSU 블록의 레이어 수 (예: RSU4는 height=4, RSU7는 height=7).
    dilation_height : int
        블록 내 컨볼루션의 dilation 값.
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, height: int, dilation_height: int):
        super(RSUBlock, self).__init__()

        self.height = height
        self.dilation_height = dilation_height

        # 인코더 경로
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        current_in_ch = in_ch
        for i in range(height):
            dilation = dilation_height * (i + 1)
            self.encoder_blocks.append(conv_block.ConvBlock(ch_in=current_in_ch, ch_out=mid_ch, dilate=dilation))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_in_ch = mid_ch

        # 병목 (bottleneck) 블록
        self.bottleneck = conv_block.ConvBlock(ch_in=mid_ch, ch_out=mid_ch, dilate=dilation_height * (height + 1))

        # 디코더 경로
        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(height):
            dilation = dilation_height * (height - i)
            self.up_layers.append(nn.ConvTranspose2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=2, stride=2))
            # 인코더의 스킵 연결을 위해 채널을 두 배로
            self.decoder_blocks.append(conv_block.ConvBlock(ch_in=mid_ch * 2, ch_out=mid_ch, dilate=dilation))

        # 출력 레이어
        self.out_conv = conv_block.ConvBlock(ch_in=mid_ch, ch_out=out_ch, dilate=dilation_height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RSUBlock의 순전파.

        Parameters
        ----------
        x : torch.Tensor
            입력 텐서. Shape: (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            출력 텐서. Shape: (batch_size, out_ch, height, width).
        """
        encoder_features = []

        # 인코더 경로
        for i in range(self.height):
            x = self.encoder_blocks[i](x)
            encoder_features.append(x)
            x = self.pool_layers[i](x)

        # 병목 (bottleneck) 처리
        x = self.bottleneck(x)

        # 디코더 경로
        for i in range(self.height):
            x = self.up_layers[i](x)
            # 인코더의 스킵 연결 특징 맵과 결합
            enc_feat = encoder_features[self.height - i - 1]
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder_blocks[i](x)

        # 출력 레이어
        x = self.out_conv(x)

        return x
