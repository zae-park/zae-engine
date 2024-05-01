from typing import Union, Iterable
import numpy as np
import torch

EPS = torch.finfo(torch.float32).eps


def np2torch(dtype: torch.dtype):
    def deco(func):
        def wrapper(*args: Union[np.ndarray, torch.Tensor, bool, int, float], **kwargs):
            if args:
                if len(args) == 1:
                    args = a if isinstance((a := args[0]), torch.Tensor) else torch.tensor(a, dtype=dtype)
                    # return func(args)
                else:
                    args = [a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=dtype) for a in args]
                    # return func(*args)
            return func(*args, **kwargs)
            # return {k: v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=dtype) for k, v in kwargs.items()}
        return wrapper
    return deco


def shape_check(func):
    def wrapper(*args, **kwargs):
        if args:
            if len(args) == 1:
                return func(*args, **kwargs)
            else:
                args = [a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=dtype) for a in args]
                # return func(*args)
        return func(*args, **kwargs)
        # return {k: v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=dtype) for k, v in kwargs.items()}
    return wrapper

