import time

from typing import Union, Iterable
import numpy as np
import torch


def np2torch(dtype: torch.dtype):
    """
    Convert numpy arrays to torch tensors with a specified dtype.

    This decorator converts all numpy array arguments of a function to torch tensors with the specified dtype.
    If an argument is already a torch tensor, it is not converted.

    Parameters
    ----------
    dtype : torch.dtype
        The desired dtype for the torch tensors.

    Returns
    -------
    func
        The decorated function with numpy array arguments converted to torch tensors.

    Examples
    --------
    >>> @np2torch(torch.float32)
    ... def example_func(x):
    ...     return x
    >>> example_func(np.array([1, 2, 3]))  # This will be converted to a torch tensor.
    """
    def deco(func):
        def wrapper(*args: Union[np.ndarray, torch.Tensor, bool, int, float], **kwargs):
            args = tuple(a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=dtype) for a in args)
            kwargs = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=dtype) for k, v in kwargs.items()}
            return func(*args, **kwargs)
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
