import torch
import torch.nn as nn
import torch.nn.functional as F


class SKConv1D(nn.Module):
    """
    Selective Kernel Convolution for 1D inputs.

    This module performs selective kernel convolutions with different kernel sizes,
    followed by a selection mechanism to fuse the features.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int, optional
        Number of output channels. If None, it will be set to the same as `ch_in`.
    kernels : list or tuple, optional
        List of kernel sizes to be used in the convolution layers. Default is (3, 5).
    out_size : int, optional
        Output size for adaptive average pooling. If None, no pooling is applied.
    ch_ratio : int, optional
        Reduction ratio for the intermediate channels in the fuse layer. Default is 2.
    stride : int, optional
        Stride size for the convolution layers. Default is 1.

    References
    ----------
    .. [1] Li, X., Wang, W., Hu, X., & Yang, J. (2019). Selective kernel networks. 
           In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 510-519).
           Available at: https://arxiv.org/abs/1903.06586
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int = None,
        kernels: list or tuple = (3, 5),
        out_size: int = None,
        ch_ratio: int = 2,
        stride: int = 1,
    ):
        super(SKConv1D, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out or ch_in
        self.kernels = kernels
        self.kernel_valid()
        self.convs = nn.ModuleList(
            [nn.Conv1d(ch_in, self.ch_out, kernel_size=k, stride=stride, padding=k // 2) for k in kernels]
        )
        self.fuse_layer = self.fuse(self.ch_out, self.ch_out // ch_ratio)
        self.scoring = nn.Conv1d(self.ch_out // ch_ratio, self.ch_out * len(kernels), kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_pool = nn.AdaptiveAvgPool1d(out_size) if out_size else nn.Identity()
        self.selection = nn.Softmax(dim=-1)

    def kernel_valid(self):
        for k in self.kernels:
            assert k % 2 == 1, "Kernel sizes should be odd for 'same' padding to be applicable."

    def fuse(self, ch_in: int, ch_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.convs]
        mixture = self.pool(sum(feats))
        fused = self.fuse_layer(mixture)
        score = self.scoring(fused).view(fused.shape[0], self.ch_out, 1, len(self.kernels))
        score = self.selection(score)
        res = sum([feats[i] * score[:, :, :, i] for i in range(len(self.kernels))])
        res = self.out_pool(res)
        return res
