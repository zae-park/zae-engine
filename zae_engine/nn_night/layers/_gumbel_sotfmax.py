import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GumbelSoftMax(torch.autograd.Function):
    """
    Re-parametric trick for categorical problems using the Gumbel-Softmax trick.

    The Gumbel-Softmax trick allows for sampling from a categorical distribution
    in a differentiable manner, which is useful for incorporating categorical variables
    into neural networks. This implementation follows the method described in [1]_.

    Methods
    -------
    forward(ctx, *args)
        Computes the forward pass, returning a rounded tensor while retaining
        differentiability.

    backward(ctx, *grad_outputs)
        Computes the backward pass, returning the gradient of the input.

    Examples
    --------
    >>> import torch
    >>> tmp = torch.rand(10, dtype=torch.float64).clone().detach().requires_grad_(True)
    >>> rounded = torch.round(tmp)
    >>> stopped = tmp.detach()
    >>> output = GumbelSoftMax.apply(tmp)
    >>> output.backward(torch.ones_like(tmp))

    References
    ----------
    .. [1] https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    """

    @staticmethod
    def forward(ctx, x):
        """
        Compute the forward pass for the Gumbel-Softmax trick.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object for storing information to be used in the backward pass.
        *args : tuple
            Expect the first argument to be a single tensor `x`.

        Returns
        -------
        torch.Tensor
            The rounded tensor while retaining differentiability.
        """

        activated = F.softmax(x, dim=-1)
        stopped = x.detach()
        ctx.save_for_backward(x, activated, stopped)
        return x + activated - stopped  # returned rounded tensor.

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Compute the backward pass for the Gumbel-Softmax trick.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object containing saved tensors from the forward pass.
        *grad_outputs : tuple
            Gradients passed from the next layer.

        Returns
        -------
        torch.Tensor
            The gradient of the input tensor.
        """
        grad = grad_outputs[0]
        x, activated, stopped = ctx.saved_tensors
        grad = torch.autograd.grad(x, x, grad)
        return grad
