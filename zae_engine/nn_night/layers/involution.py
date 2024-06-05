import torch.nn as nn
import torch.nn.functional as F


class Inv1d(nn.Module):
    """The involution layer for 1D input [1]_.

    ...
    Parameters
    ----------
    ch : int
        Number of channels in the input tensor (signal or 1D arr)
    num_groups : int
        Number of channels produced by the convolution
    kernel_size : int
        Size of the convolving kernel
    stride : int
        Stride of the convolution. Default: 1
    reduction_ratio : int
        Ratio of channel reduction. This value must be divisor of ch.

    Returns
    -------
    tensor : tensor

    References
    ----------
    .. [1]  Li, Duo, et al.
            "Involution: Inverting the inherence of convolution for visual recognition."
            Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
    """

    def __init__(self, ch: int, num_groups: int, kernel_size: int, stride: int, reduction_ratio: int):
        super(Inv1d, self).__init__()
        self.ch = ch
        self.num_groups = num_groups
        self.group = ch // num_groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        self.k_gen = self.kernel_generator()
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), padding=(0, (kernel_size - 1) // 2), stride=(1, stride))

        assert ch % num_groups == 0

    def kernel_generator(self):
        conv1 = nn.Conv1d(self.ch, self.ch // self.reduction_ratio, kernel_size=(1,))
        conv2 = nn.Conv1d(self.ch // self.reduction_ratio, self.kernel_size * self.num_groups, kernel_size=(1,))
        return nn.Sequential(*[conv1, nn.ReLU(), nn.BatchNorm1d(self.ch // self.reduction_ratio), conv2])

    def forward(self, x):
        b, ch, dim = x.shape
        assert ch == self.ch
        assert dim % self.stride == 0
        out_dim = dim // self.stride
        unfolded = self.unfold(x.unsqueeze(2))
        unfolded = unfolded.view(b, self.num_groups, self.group, self.kernel_size, out_dim)
        pooled = F.adaptive_max_pool1d(x, out_dim)
        kernel = self.k_gen(pooled).view(b, self.num_groups, self.kernel_size, out_dim).unsqueeze(2)
        out = kernel * unfolded
        return out.sum(dim=3).view(b, ch, out_dim)
