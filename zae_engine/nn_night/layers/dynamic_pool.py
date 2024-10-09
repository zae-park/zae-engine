import torch
import torch.nn as nn

from . import _gumbel_sotfmax


class DynOPool(nn.Module):
    """
    Dynamic Pooling Layer using Gumbel Softmax trick for discrete pooling ratios.

    This layer dynamically adjusts the pooling ratio using a learnable parameter,
    allowing for adaptive pooling during training. The Gumbel Softmax trick is
    applied to ensure the ratio remains discrete.

    Reference:
        - DynOPool: Dynamic Optimization Pooling, https://arxiv.org/abs/2205.15254

    Attributes
    ----------
    ratio : nn.Parameter
        Learnable parameter representing the pooling ratio.
    trick : function
        Function to apply the Gumbel Softmax trick.

    Methods
    -------
    bilinear_interpolation(x, to_dim)
        Performs bilinear interpolation on the input tensor to the specified dimension.
    forward(x)
        Forward pass for the DynOPool layer.
    """

    def __init__(self):
        super(DynOPool, self).__init__()
        self.ratio = nn.Parameter(torch.empty(1, dtype=torch.float32, requires_grad=True))
        nn.init.constant_(self.ratio, 1)
        self.trick = _gumbel_sotfmax.GumbelSoftMax.apply

    def bilinear_interpolation(self, x, to_dim):
        """
        Perform bilinear interpolation on the input tensor to the specified dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, depth).
        to_dim : torch.Tensor
            Target dimension after interpolation.

        Returns
        -------
        torch.Tensor
            Interpolated tensor.
        """
        b, c, d = x.shape
        d_ = to_dim

        idx = torch.arange(d_.item()).to(x.device)
        indices = (torch.cat((idx + 0.25, idx + 0.75), dim=-1) * self.ratio).long()
        # indices = torch.clamp(indices, 0, d - 1)  # Ensure indices are within bounds
        indices_ = indices.repeat(b, c, 1)
        sampled = torch.gather(x, -1, indices_).reshape(b, c, -1, 2)
        return sampled.mean(-1)

    def forward(self, x):
        """
        Forward pass for the DynOPool layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, depth).

        Returns
        -------
        torch.Tensor
            Pooled tensor.
        """
        b, c, d = x.shape
        tricked = self.trick(d * self.ratio)
        res = self.bilinear_interpolation(x, tricked)
        return res
