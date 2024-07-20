import torch
import torch.nn.functional as F


class GumbelSoftMax(torch.autograd.Function):
    """
    Re-parametric trick for categorical problems using the Gumbel-Softmax trick.

    The Gumbel-Softmax trick allows for sampling from a categorical distribution
    in a differentiable manner, which is useful for incorporating categorical variables
    into neural networks. This implementation follows the method described in [1]_ and [2]_.

    Methods
    -------
    forward(ctx, logits, temperature)
        Computes the forward pass using the Gumbel-Softmax trick.

    backward(ctx, grad_output)
        Computes the backward pass, returning the gradient of the input.

    Examples
    --------
    >>> import torch
    >>> logits = torch.rand(10, 3, dtype=torch.float64).clone().detach().requires_grad_(True)
    >>> temperature = 1.0
    >>> output = GumbelSoftMax.apply(logits, temperature)
    >>> output.backward(torch.ones_like(output))

    References
    ----------
    .. [1] https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    .. [2] Eric Jang, Shixiang Gu, and Ben Poole. "Categorical Reparameterization with Gumbel-Softmax."
           In International Conference on Learning Representations (ICLR), 2017. https://arxiv.org/abs/1611.01144
    """

    @staticmethod
    def forward(ctx, logits, temperature):
        """
        Compute the forward pass for the Gumbel-Softmax trick.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object for storing information to be used in the backward pass.
        logits : torch.Tensor
            Input logits for the categorical distribution.
        temperature : float
            Temperature parameter for the Gumbel-Softmax distribution.

        Returns
        -------
        torch.Tensor
            Sampled tensor from the Gumbel-Softmax distribution.
        """
        gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
        gumbel_logits = (logits + gumbels) / temperature
        y_soft = F.softmax(gumbel_logits, dim=-1)

        ctx.save_for_backward(y_soft)
        return y_soft

    @staticmethod
    def backward(ctx, grad_output):
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
        (y_soft,) = ctx.saved_tensors
        grad_input = y_soft * (grad_output - (grad_output * y_soft).sum(dim=-1, keepdim=True))
        return grad_input, None  # return None for temperature since it's a constant
