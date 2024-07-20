import torch


class GumbelSoftMax(torch.autograd.Function):
    """
    Re-parametric trick for categorical problems using the Gumbel-Softmax trick.

    The Gumbel-Softmax trick allows for sampling from a categorical distribution
    in a differentiable manner, which is useful for incorporating categorical variables
    into neural networks. This implementation follows the method described in [1]_ and [2]_.

    Methods
    -------
    forward(ctx, x)
        Computes the forward pass, returning a rounded tensor while retaining
        differentiability.

    backward(ctx, grad_output)
        Computes the backward pass, returning the gradient of the input.

    Examples
    --------
    >>> import torch
    >>> tmp = torch.rand(10, dtype=torch.float64).clone().detach().requires_grad_(True)
    >>> output = GumbelSoftMax.apply(tmp)
    >>> output.backward(torch.ones_like(tmp))

    References
    ----------
    .. [1] https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    .. [2] Eric Jang, Shixiang Gu, and Ben Poole. "Categorical Reparameterization with Gumbel-Softmax."
           In International Conference on Learning Representations (ICLR), 2017. https://arxiv.org/abs/1611.01144
    """

    @staticmethod
    def forward(ctx, x):
        """
        Compute the forward pass for the Gumbel-Softmax trick.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object for storing information to be used in the backward pass.
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The rounded tensor while retaining differentiability.
        """
        rounded = torch.round(x)
        stopped = x.detach()
        ctx.save_for_backward(x, rounded, stopped)
        return x + rounded - stopped

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Compute the backward pass for the Gumbel-Softmax trick.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object containing saved tensors from the forward pass.
        grad_output : torch.Tensor
            Gradient passed from the next layer.

        Returns
        -------
        torch.Tensor
            The gradient of the input tensor.
        """
        return grad_output
