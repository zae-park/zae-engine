import time

from typing import Union, Iterable
import numpy as np
import torch


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


def shape_check(*keys):
    if len(keys) == 1:
        keys = keys[0]
        assert isinstance(keys, int), f"Input the num of args to check"
        assert keys > 1, f"Cannot compare shape of single argument"
    else:
        for k in keys:
            assert isinstance(k, str), f"Input the multiple arg strings to check"

    def deco(func):
        def wrapper(*args, **kwargs):
            if isinstance(keys, int):
                shape_list = [a.shape for a in args[:keys]]
            else:
                shape_list = [kwargs[key].shape for key in keys]
            print(shape_list)
            assert len(set(shape_list)) == 1, "Shape of given args is not same."
            return func(*args, **kwargs)

        return wrapper

    return deco


def tictoc(func):
    def wrapper(*args, **kwargs):
        kickoff = time.time()
        out = func(*args, **kwargs)
        print(f"Elapsed time [sec]: {time.time() - kickoff}")
        return out

    return wrapper
