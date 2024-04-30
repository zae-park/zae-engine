import numpy as np
import torch


def np2torch(dtype: torch.dtype):
    def deco(func):
        def wrapper(x: np.ndarray | torch.Tensor):
            x = torch.tensor(x, dtype=dtype) if isinstance(x, np.ndarray) else x.astype(dtype)
            return func(x)
        return wrapper
    return deco
