import torch
import torch.nn as nn

from torch.nn.common_types import _size_1_t


class SE1d(nn.Module):
    """
    Generalized version of Squeeze and Excitation module (https://arxiv.org/abs/1709.01507).
    If spatial argument is False, this work equivalent to SE module.
    :param ch: int
        The channels-wise dimension of input tensor.
    :param reduction: int
        The Squeeze and Excitation ratio. Note that it must be a divisor of ch.
    :param bias: bool
    """

    def __init__(self, ch_in: int, reduction: int = 8, bias: bool = False, *args, **kwargs):
        super(SE1d, self).__init__()
        self.reduction = reduction
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        assert ch_in % reduction == 0, f'Received invalid arguments. The "reduction" must be a divisor of "B".'
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(ch_in, ch_in // reduction, kernel_size=(1,), bias=bias)
        self.fc2 = nn.Conv1d(ch_in // reduction, ch_in, kernel_size=(1,), bias=bias)

    def channel_wise(self, x):
        vec = self.relu(self.fc(self.pool(x)))
        vec = self.sigmoid(self.fc2(vec))
        return vec

    def forward(self, x):
        return x * self.channel_wise(x)


class CBAM1d(nn.Module):
    """
    Generalized version of Convolutional Block Attention Module (https://arxiv.org/abs/1807.06521).
    Inherit SE module for channel-wise attention, and operates spatial-wise attention serially.
    The spatial-wise operation consists of pooling and channel aggregation.
    If `conv_pool` option is False, channel aggregation works with MAX & AVG pooled tensor.
    If `conv_pool` option is True, channel aggregation works with parametric pooled (convolution).

    :param ch: int
        The channels-wise dimension of input tensor.
    :param reduction: int
        The Squeeze and Excitation ratio. Note that it must be a divisor of ch.
    :param bias: bool
    :param conv_pool: bool
    """

    def __init__(
        self,
        ch_in: int,
        kernel_size: _size_1_t = 7,
        reduction: int = 8,
        bias: bool = False,
        conv_pool: bool = False,
        *args,
        **kwargs,
    ):
        super(CBAM1d, self).__init__()
        self.kernel_size = kernel_size
        self.conv_pool = conv_pool

        self.se_module = SE1d(ch_in=ch_in, reduction=reduction, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.ch_pool = nn.Conv1d(
            in_channels=ch_in if conv_pool else 2, out_channels=1, kernel_size=kernel_size, bias=bias
        )

    def spatial_wise(self, x: torch.Tensor):
        if self.conv_pool:
            attn_map = self.ch_pool(x)
        else:
            max_map = torch.max(x, dim=1, keepdim=True)
            avg_map = torch.mean(x, dim=1, keepdim=True)
            attn_map = self.ch_pool(torch.cat((max_map, avg_map), dim=1))

        return self.sigmoid(attn_map)

    def forward(self, x):
        x *= self.se_module(x)
        return x * self.spatial_wise(x)
