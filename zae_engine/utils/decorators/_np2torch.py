import inspect
from typing import Union, Callable
import numpy as np
import torch


def np2torch_fn(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            new_args = list(args)
            if keys:
                for key in keys:
                    if key in new_args[0]:
                        new_args[0][key] = torch.tensor(new_args[0][key], dtype=dtype)
            else:
                for i in range(n or len(new_args)):
                    if isinstance(new_args[i], np.ndarray):
                        new_args[i] = torch.tensor(new_args[i], dtype=dtype)
            return func(*new_args, **kwargs)

        return wrapper

    return decorator


def np2torch_mtd(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            new_args = list(args)
            if keys:
                for key in keys:
                    if key in new_args[0]:
                        new_args[0][key] = torch.tensor(new_args[0][key], dtype=dtype)
            else:
                for i in range(n or len(new_args)):
                    if isinstance(new_args[i], np.ndarray):
                        new_args[i] = torch.tensor(new_args[i], dtype=dtype)
            return func(self, *new_args, **kwargs)

        return wrapper

    return decorator


def np2torch(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
    """
    Convert numpy arrays to torch tensors with a specified dtype.
    This decorator automatically detects if it is used in a class method or a standalone function and behaves accordingly.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.

    Parameters
    ----------
    dtype : torch.dtype
        The desired dtype for the torch tensors.
    n : int, optional
        The number of initial arguments to convert. If None, all numpy array arguments are converted.
    keys : str, optional
        The keys of the dictionary to convert. If None, all arguments are converted.

    Returns
    -------
    func
        The decorated function with numpy array arguments converted to torch tensors.

    Examples
    --------
    >>> @np2torch(torch.float32, n=2)
    ... def example_func(x, y, z):
    ...     return x, y, z
    >>> example_func(np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]))
    # This will convert only the first two numpy arrays to torch tensors.

    >>> class Example:
    ...     @np2torch(torch.float32, "x", "y")
    ...     def example_method(self, batch):
    ...         return batch
    >>> example = Example()
    >>> example.example_method({"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])})
    # This will convert the values of 'x' and 'y' keys in the dictionary to torch tensors.
    """

    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if len(args) > 0 and hasattr(args[0], func.__name__):
                # 메소드인 경우
                decorator = np2torch_mtd(dtype, *keys, n=n)
                return decorator(func)(*args, **kwargs)
            else:
                # 함수인 경우
                decorator = np2torch_fn(dtype, *keys, n=n)
                return decorator(func)(*args, **kwargs)

        return wrapper

    return deco
