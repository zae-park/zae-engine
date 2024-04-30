import torch


class GumbelSoftMax(torch.autograd.Function):
    """
    Re-parametric trick for categorical problems.
    ref: https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    TODO:
        - fix description
        - add test case
        - optimize
    """

    @staticmethod
    def forward(ctx, *args):
        # Expect input argument to be a single tensor.
        x = args[0]
        rounded = torch.round(x)
        stopped = x.detach()
        ctx.save_for_backward(x, rounded, stopped)
        return x + rounded - stopped  # returned rounded tensor.

    @staticmethod
    def backward(ctx, *grad_outputs):
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
