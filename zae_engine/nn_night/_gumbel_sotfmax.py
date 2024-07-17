import torch


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

    # TODO: add test case & optimization
    @staticmethod
    def forward(ctx, *args):
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
        # Expect input argument to be a single tensor.
        x = args[0]
        rounded = torch.round(x)
        stopped = x.detach()
        ctx.save_for_backward(x, rounded, stopped)
        return x + rounded - stopped  # returned rounded tensor.

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
        # return received gradient as is.
        # Assume that the gradient of the rounded tensor has the same as that of the input.
        return grad


if __name__ == "__main__":

    tmp = torch.rand(10, dtype=torch.float64).clone().detach().requires_grad_(True)
    r = torch.round(tmp)
    sg = tmp.detach()
    torch.autograd.gradcheck(GumbelRound.apply, (tmp, r, sg))
    torch.autograd.gradgradcheck(GumbelRound.apply, (tmp, r, sg))
