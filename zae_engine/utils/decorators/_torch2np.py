from typing import Union, Callable, Type
import numpy as np
import torch


def getter(ts: torch.Tensor) -> np.ndarray:
    return ts.detach().numpy()


def torch2np_fn(dtype: np.dtype, *keys: str, n: int = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            new_args = list(args)
            if keys:
                for key in keys:
                    v = new_args[0][key]
                    new_args[0][key] = getter(v).astype(dtype) if isinstance(v, torch.Tensor) else v

            else:
                for i in range(n or len(new_args)):
                    if isinstance(new_args[i], torch.Tensor):
                        new_args[i] = getter(new_args[i]).astype(dtype)
            return func(*new_args, **kwargs)

        return wrapper

    return decorator


def torch2np_mtd(dtype: np.dtype, *keys: str, n: int = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            new_args = list(args)
            if keys:
                for key in keys:
                    v = new_args[0][key]
                    new_args[0][key] = getter(v).astype(dtype) if isinstance(v, torch.Tensor) else v

            else:
                for i in range(n or len(new_args)):
                    if isinstance(new_args[i], torch.Tensor):
                        new_args[i] = getter(new_args[i]).astype(dtype)
            return func(self, *new_args, **kwargs)

        return wrapper

    return decorator


def torch2np(dtype: Type[np.dtype], *keys: str, n: int = None) -> Callable:
    """
    Convert torch tensors to numpy arrays with a specified dtype.
    This decorator automatically detects if it is used in a class method or a standalone function and behaves accordingly.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.

    Parameters
    ----------
    dtype : np.dtype
        The desired dtype for the numpy arrays.
    n : int, optional
        The number of initial arguments to convert. If None, all torch tensor arguments are converted.
    keys : str, optional
        The keys of the dictionary to convert. If None, all arguments are converted.

    Returns
    -------
    func
        The decorated function with torch tensor arguments converted to numpy arrays.

    Examples
    --------
    >>> @torch2np(np.float32, n=2)
    ... def example_func(x, y, z):
    ...     return x, y, z
    >>> example_func(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9]))
    # This will convert only the first two torch tensors to numpy arrays.

    >>> class Example:
    ...     @torch2np(np.float32, "x", "y")
    ...     def example_method(self, batch):
    ...         return batch
    >>> example = Example()
    >>> example.example_method({"x": torch.tensor([1, 2, 3]), "y": torch.tensor([4, 5, 6]), "z": torch.tensor([7, 8, 9])})
    # This will convert the values of 'x' and 'y' keys in the dictionary to numpy arrays.
    """

    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if len(args) > 0 and hasattr(args[0], func.__name__):
                # 메소드인 경우
                decorator = torch2np_mtd(dtype, *keys, n=n)
                return decorator(func)(*args, **kwargs)
            else:
                # 함수인 경우
                decorator = torch2np_fn(dtype, *keys, n=n)
                return decorator(func)(*args, **kwargs)

        return wrapper

    return deco
