import torch
import torch.nn as nn


class Additional(nn.ModuleList):
    """
    Additional connection module.

    This module extends nn.ModuleList to implement an additional connection.
    Each input tensor is passed through its corresponding module,
    and the output tensors are summed. If the shapes of the output tensors
    do not match, an error is raised.

    Parameters
    ----------
    *args : nn.Module
        Sequence of PyTorch modules. Each module will be applied to
        a corresponding input tensor in the forward pass.

    Methods
    -------
    forward(*inputs)
        Applies each module to its corresponding input tensor and returns
        the sum of the output tensors. If the shapes of the output tensors
        do not match, an error is raised.
    """

    def __init__(self, *args):
        """
        Initialize the Additional module with a sequence of sub-modules.

        Parameters
        ----------
        *args : nn.Module
            Sequence of PyTorch modules to be applied to the input tensors.
        """
        super(Additional, self).__init__(args)

    def forward(self, *inputs):
        """
        Forward pass through the additional block.

        Applies each module to its corresponding input tensor and returns
        the sum of the output tensors. If the shapes of the output tensors
        do not match, an error is raised.

        Parameters
        ----------
        *inputs : torch.Tensor
            Sequence of input tensors. Each tensor is passed through its corresponding module.

        Returns
        -------
        torch.Tensor
            The sum of the output tensors of each module.

        Raises
        ------
        ValueError
            If the output tensors have mismatched shapes.
        """

        if len(inputs) != len(self):
            raise ValueError(f"Expected {len(self)} input tensors, but got {len(inputs)}.")

        # Apply each module to its corresponding input and store the outputs in a list
        outputs = [layer(inputs[i]) for i, layer in enumerate(self)]

        # Ensure that all output tensors have the same shape
        first_shape = outputs[0].shape
        for output in outputs[1:]:
            if output.shape != first_shape:
                raise ValueError(f"Shape mismatch: expected {first_shape}, but got {output.shape}")

        # Return the sum of the output tensors
        return sum(outputs)
