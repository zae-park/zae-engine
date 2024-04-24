import torch
import torch.nn as nn

from .. import _backward


class DynOPool(nn.Module):
    """
    https://arxiv.org/abs/2205.15254
    """

    def __init__(self):
        super(DynOPool, self).__init__()
        self.ratio = nn.Parameter(torch.empty(1, dtype=torch.float32, requires_grad=True))
        nn.init.constant_(self.ratio, 1)
        self.trick = _backward.GumbelRound.apply

    def bilinear_interpolation(self, x, to_dim):
        b, c, d = x.shape
        d_ = to_dim

        idx = torch.arange(d_.item()).to(x.device)
        indices = (torch.cat((idx + 0.25, idx + 0.75), dim=-1) * self.ratio).long()
        indices_ = indices.repeat(b, c, 1)
        sampled = torch.gather(x, -1, indices_).reshape(b, c, -1, 2)
        return sampled.mean(-1)

    def forward(self, x):
        b, c, d = x.shape
        tricked = self.trick(d * self.ratio)
        res = self.bilinear_interpolation(x, tricked)
        return res
