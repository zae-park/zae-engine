import torch
import torch.nn as nn
import torch.nn.functional as F


class SKConv1D(nn.Module):
    """
    TODO:
        - fill description
        - add test case
        - optimize
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int = None,
        kernels: list or tuple = (3, 5),
        out_size=None,
        ch_ratio: int = 2,
        stride: int = 1,
        flatten=False,
    ):
        super(SKConv1D, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out or ch_out
        self.kernels = kernels
        self.kernel_valid()
        self.convs = nn.ModuleList(
            [nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=(stride,), padding="same") for k in kernels]
        )
        self.fuse_layer = self.fuse(ch_out, ch_out // ch_ratio)
        self.scoring = nn.Conv1d(ch_out // ch_ratio, ch_out * len(kernels), kernel_size=(1,))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_pool = nn.AdaptiveAvgPool1d(out_size) if out_size else nn.Identity()
        self.selection = nn.Softmax(-1)
        self.flatten = flatten

    def kernel_valid(self):
        for k in self.kernels:
            assert k % 2

    def fuse(self, ch_in, ch_out):
        return nn.Sequential(*[nn.Conv1d(ch_in, ch_out, kernel_size=(1,)), nn.BatchNorm1d(ch_out), nn.ReLU()])

    def forward(self, x):
        feats = [c(x) for c in self.convs]
        if self.flatten == "flat1":
            res = torch.cat(feats, dim=-1)
        else:
            mixture = self.pool(sum(feats))
            fused = self.fuse_layer(mixture)
            score = self.scoring(fused).reshape(fused.shape[0], self.ch_out, 1, len(self.kernels))
            score = self.selection(score)
            if self.flatten == "flat2":
                res = torch.cat([feats[i] * score[:, :, :, i] for i in range(len(self.kernels))], dim=-1)
            else:
                res = sum([feats[i] * score[:, :, :, i] for i in range(len(self.kernels))])
        res = self.out_pool(res) if self.out_pool else res
        return res
