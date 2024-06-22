import torch.nn as nn


class ClippedReLU(nn.Module):
    """
    A modified version of the ReLU activation function.

    The input tensor is bounded to the range [lower, upper] using two ReLU functions.

    Parameters
    ----------
    upper : float, optional
        The upper threshold, by default 1.0.
    lower : float, optional
        The lower threshold, by default 0.0.

    Methods
    -------
    forward(x : torch.Tensor) -> torch.Tensor
        Applies the ClippedReLU function to the input tensor.

    Examples
    --------
    >>> import torch
    >>> from clipped_relu import ClippedReLU
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
    >>> clipped_relu = ClippedReLU(upper=1.0, lower=0.0)
    >>> clipped_relu(x)
    tensor([0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000])
    """

    def __init__(self, upper: float = 1.0, lower: float = 0.0):
        super(ClippedReLU, self).__init__()
        assert upper > lower, (
            f'Expect 0th argument "upper" greater than 1st argument "lower".'
            f" But upper({upper:.2f}) is less than lower ({lower:.2f})"
        )
        self.upper = upper
        self.lower = lower
        self.upper_act = nn.ReLU()
        self.lower_act = nn.ReLU()

    def forward(self, x):
        """
        Apply the ClippedReLU function to the input tensor.

        The function is defined as:
        f(x) = min(max(x, lower), upper)

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the ClippedReLU function.
        """
        return self.lower + self.upper - x + self.lower_act(x - self.lower) - self.upper_act(self.upper - x)
