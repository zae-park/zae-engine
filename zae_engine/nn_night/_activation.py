import torch.nn as nn


class ClippedReLU(nn.Module):
    """
    The modified version of ReLU.
    Input tensor bounded to the range [lower, upper].
    :param upper: float
        The upper threshold. default is 1.
    :param lower: float
        The lower threshold. default is 0.
    """

    def __init__(self, upper: float = 1.0, lower: float = 0.0):
        super(ClippedReLU, self).__init__()
        assert upper > lower, (
            f'Expect 0th argument "upper" greater than 1st argument "lower".'
            f" But upper({upper:.2f}) is less than lower ({lower:.2f})"
        )
        self.upper = upper
        self.lower = lower
        self.act = nn.ReLU()

    def forward(self, x):
        return self.lower + self.upper - x + self.act(x - self.lower) - self.act(self.upper - x)
