from typing import Callable, Type
import numpy as np
import torch


def getter(ts: torch.Tensor) -> np.ndarray:
    return ts.detach().numpy()


def torch2np(dtype: np.dtype, *keys: str, n: int = None) -> Callable:
    """
    Convert torch tensors to numpy arrays with a specified dtype.
    This decorator handles both class methods and standalone functions.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.
    If n is specified, only the first n arguments (after 'self' for methods) are converted.
    If 'n' is greater than the number of arguments, conversion proceeds without error.

    Parameters
    ----------
    dtype : np.dtype
        The desired dtype for the numpy arrays.
    *keys : str, optional
        The keys of the dictionary to convert. If not provided, arguments are converted based on 'n'.
    n : int, optional
        The number of initial arguments to convert. If None, all torch tensor arguments are converted.
        If 'n' is greater than the number of arguments, conversion proceeds without error.

    Returns
    -------
    Callable
        The decorated function with torch tensor arguments converted to numpy arrays.

    Examples
    --------
    >>> @torch2np(np.float32, n=2)
    ... def example_func(x, y, z):
    ...     return x, y, z
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([4, 5, 6])
    >>> z = torch.tensor([7, 8, 9])
    >>> example_func(x, y, z)
    # This will convert only the first two torch tensors to numpy arrays.

    >>> class Example:
    ...     @torch2np(np.float32, "x", "y")
    ...     def example_method(self, batch):
    ...         return batch
    >>> example = Example()
    >>> batch = {"x": torch.tensor([1, 2, 3]), "y": torch.tensor([4, 5, 6]), "z": torch.tensor([7, 8, 9])}
    >>> example.example_method(batch)
    # This will convert the values of 'x' and 'y' keys in the dictionary to numpy arrays.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            new_args = list(args)

            # Determine if the function is a method by checking if the first argument has the function as an attribute
            if len(args) > 0 and hasattr(args[0], func.__name__):
                # Probably a method; skip 'self'
                offset = 1
            else:
                offset = 0

            if keys:
                # Assuming new_args[offset] is a dict
                for key in keys:
                    v = new_args[offset][key]
                    if isinstance(v, torch.Tensor):
                        new_args[offset][key] = getter(v).astype(dtype)
            else:
                # Adjust end_index to avoid IndexError if n is greater than the number of arguments
                end_index = min(offset + n, len(new_args)) if n is not None else len(new_args)
                for i in range(offset, end_index):
                    if isinstance(new_args[i], torch.Tensor):
                        new_args[i] = getter(new_args[i]).astype(dtype)

            return func(*new_args, **kwargs)

        return wrapper

    return decorator
