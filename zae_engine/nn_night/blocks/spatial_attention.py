import torch
import torch.nn as nn

from torch.nn.common_types import _size_1_t


class SE1d(nn.Module):
    """
    Squeeze and Excitation module for 1D inputs.

    This module implements the Squeeze and Excitation (SE) block.

    Parameters
    ----------
    ch_in : int
        The channel-wise dimension of the input tensor.
    reduction : int, optional
        The reduction ratio for the SE block. Default is 8. Must be a divisor of ch_in.
    bias : bool, optional
        Whether to use bias in the fully connected layers. Default is False.

    References
    ----------
    .. [1] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
           In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
           https://arxiv.org/abs/1709.01507
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
    Convolutional Block Attention Module for 1D inputs.

    This module implements the Convolutional Block Attention Module (CBAM).

    Parameters
    ----------
    ch_in : int
        The channel-wise dimension of the input tensor.
    kernel_size : _size_1_t, optional
        The kernel size for the convolutional layer. Default is 7.
    reduction : int, optional
        The reduction ratio for the SE block. Default is 8. Must be a divisor of ch_in.
    bias : bool, optional
        Whether to use bias in the convolutional and fully connected layers. Default is False.
    conv_pool : bool, optional
        If True, use convolutional pooling for the spatial attention mechanism. Default is False.

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module.
           In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
           https://arxiv.org/abs/1807.06521
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
            in_channels=ch_in if conv_pool else 2,
            out_channels=1,
            kernel_size=kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

    def spatial_wise(self, x: torch.Tensor):
        if self.conv_pool:
            attn_map = self.ch_pool(x)
        else:
            max_map = torch.max(x, dim=1, keepdim=True)[0]
            avg_map = torch.mean(x, dim=1, keepdim=True)
            attn_map = self.ch_pool(torch.cat((max_map, avg_map), dim=1))

        return self.sigmoid(attn_map)

    def forward(self, x):
        x *= self.se_module(x)
        return x * self.spatial_wise(x)
