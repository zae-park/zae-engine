import torch.nn as nn


class Residual(nn.Sequential):
    """
    Residual connection.
    As nn.Sequential, receive modules but connect the output tensor to an input tensor.
    """

    def __init__(self, *args):
        super(Residual, self).__init__(*args)

    def forward(self, x):
        """
        The 'module' is sequence of arguments provides in __init__.
        :param x: Input tensor
        :return: Sum of input tensor and output of sequence.
        """

        residual = x
        for module in self:
            x = module(x)
        return x + residual
