from typing import Callable, TypeVar, Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.common_types as types


class ConvBlock(nn.Module):
    kernel_type = Union[int, Tuple[int, int]]

    """
    Residual Block with Convolution, Batch Normalization, and ReLU Activation.

    This module performs a convolution followed by batch normalization and ReLU activation.
    It serves as a fundamental building block in the RSU (Residual U-block) structure.

    Parameters
    ----------
    ch_in : int, optional
        Number of input channels. Default is 3.
    ch_out : int, optional
        Number of output channels. Default is 3.
    dilate : int, optional
        Dilation rate for the convolution. Default is 1.
    """

    def __init__(
        self,
        ch_in: int = 3,
        ch_out: int = 3,
        kernel_size: kernel_type = 3,
        dilate: int = 1,
        pre_norm: bool = False,
        conv_layer: nn.Module = nn.Conv2d,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super(ConvBlock, self).__init__()
        self.pre_norm = pre_norm
        self.kernel_size = kernel_size
        self.conv = conv_layer(ch_in, ch_out, kernel_size=3, padding=1 * dilate, dilation=1 * dilate)
        self.norm = norm_layer(ch_out)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvBlock block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channel_in, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, channel_out, height, width).
        """
        if self.pre_norm:
            return self.act(self.conv(self.norm(x)))
        else:
            return self.act(self.norm(self.conv(x)))
