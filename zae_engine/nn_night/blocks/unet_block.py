from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from . import resblock


class UNetBlock(resblock.BasicBlock):
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
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = out if self.downsample is None else self.downsample(out)
        return out
