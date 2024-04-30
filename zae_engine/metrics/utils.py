from typing import Union, Iterable
import numpy as np
import torch


def np2torch(dtype: torch.dtype):
    def deco(func):
        def wrapper(*args: Union[np.ndarray, torch.Tensor, bool, int, float]):

            if len(args) == 1:
                args = a if isinstance((a := args[0]), torch.Tensor) else torch.tensor(a, dtype=dtype)
                return func(args)
            else:
                args = [a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=dtype) for a in args]
                return func(*args)
        return wrapper
    return deco
