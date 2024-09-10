import torch
import torch.nn as nn


class Additional(nn.Module):
    """
    Additional connection module.

    This module extends nn.Module and uses nn.ModuleList to implement an additional connection.
    The output is the sum of the results of applying a sequence of modules to the input tensor.

    Parameters
    ----------
    *args : nn.Module
        Sequence of PyTorch modules to be applied to the input tensor.

    Methods
    -------
    forward(x)
        Applies the sequence of modules to the input tensor and returns the sum of their outputs.
    """

    def __init__(self, *args: nn.Module):
        """
        Initialize the AdditionalLayer with a sequence of sub-modules.

        Parameters
        ----------
        *args : nn.Module
            Sequence of PyTorch modules to be applied to the input tensor.
        """
        super(Additional, self).__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x):
        """
        Forward pass through the additional block.

        Applies the sequence of modules to the input tensor and returns the sum of the outputs of the modules.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The sum of the outputs of the sequence of modules.
        """
        output_sum = 0
        for layer in self.layers:
            output_sum += layer(x)
        return output_sum
