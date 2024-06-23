import torch.nn as nn


class Residual(nn.Sequential):
    """
    Residual connection module.

    This module extends nn.Sequential to implement a residual connection. The input tensor is added
    to the output tensor of the sequence of modules provided during initialization, similar to a
    residual block in ResNet architectures.

    Parameters
    ----------
    *args : nn.Module
        Sequence of PyTorch modules to be applied to the input tensor.

    Methods
    -------
    forward(x)
        Applies the sequence of modules to the input tensor and returns the sum of the input tensor
        and the output tensor.
    """

    def __init__(self, *args):
        """
        Initialize the Residual module with a sequence of sub-modules.

        Parameters
        ----------
        *args : nn.Module
            Sequence of PyTorch modules to be applied to the input tensor.
        """
        super(Residual, self).__init__(*args)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Applies the sequence of modules to the input tensor and returns the sum of the input tensor
        and the output tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The sum of the input tensor and the output of the sequence of modules.
        """

        residual = x
        for module in self:
            x = module(x)
        return x + residual
