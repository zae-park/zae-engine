from typing import Callable

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic residual block.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    stride : int, optional
        Stride of the convolution. Default is 1.
    groups : int, optional
        Number of groups for the convolution. Default is 1.
    dilation : int, optional
        Dilation for the convolution. Default is 1.
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use. Default is nn.BatchNorm2d.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """

    expansion: int = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        assert (ch_in % groups) == (ch_out % groups) == 0, "Group must be common divisor of ch_in and ch_out."
        self.ch_in = ch_in
        self.ch_out = ch_out

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.norm1 = norm_layer(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding="same", groups=groups)
        self.norm2 = norm_layer(ch_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or ch_in != ch_out * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, stride=stride),
                norm_layer(ch_out * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        identity = x if self.downsample is None else self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    stride : int, optional
        Stride of the convolution. Default is 1.
    groups : int, optional
        Number of groups for the convolution. Default is 1.
    dilation : int, optional
        Dilation for the convolution. Default is 1.
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use. Default is nn.BatchNorm2d.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """

    expansion: int = 4

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        assert (ch_in % groups) == (ch_out % groups) == 0, "Group must be common divisor of ch_in and ch_out."
        self.ch_in = ch_in
        self.ch_out = ch_out

        width = ch_out * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1)
        self.norm1 = norm_layer(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, width, kernel_size=3, stride=stride, padding=1, groups=groups, dilation=dilation)
        self.norm2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width, ch_out * self.expansion, kernel_size=1, stride=1)
        self.norm3 = norm_layer(ch_out * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.stride = stride

        if stride != 1 or ch_in != ch_out * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, stride=stride),
                norm_layer(ch_out * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        identity = x if self.downsample is None else self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out
