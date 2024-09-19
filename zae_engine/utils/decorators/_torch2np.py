from typing import Callable, Optional, Any, Dict
import numpy as np
import torch
from functools import wraps


def getter(ts: torch.Tensor) -> np.ndarray:
    """
    Convert a torch.Tensor to a numpy.ndarray.
    Ensures the tensor is detached and moved to CPU before conversion.
    """
    return ts.detach().cpu().numpy()


def torch2np(dtype: np.dtype, *keys: str, n: Optional[int] = None) -> Callable:
    """
    Convert torch tensors to numpy arrays with a specified dtype.
    This decorator handles both class methods and standalone functions.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.
    If n is specified, only the first n positional arguments (after 'self' for methods) are converted.
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
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Determine if this is a method by checking if the first argument has the function as an attribute
            if len(args) > 0 and hasattr(args[0], func.__name__):
                # Probably a method; skip 'self'
                offset = 1
            else:
                offset = 0

            modified_args = list(args)

            # If keys are specified, convert only those keys in the specified dict
            if keys:
                # Assume that the argument at 'offset' is a dict
                if offset >= len(modified_args):
                    raise ValueError(
                        f"Expected a dictionary at argument position {offset} to convert keys {keys}, but got fewer arguments."
                    )
                target_arg = modified_args[offset]
                if not isinstance(target_arg, dict):
                    raise TypeError(
                        f"Expected a dictionary at argument position {offset} to convert keys {keys}, but got {type(target_arg)}."
                    )
                for key in keys:
                    # Directly access the key without checking existence to raise KeyError if missing
                    v = target_arg[key]  # This will raise KeyError if key is missing
                    if isinstance(v, torch.Tensor):
                        target_arg[key] = getter(v).astype(dtype)
            elif n is not None:
                # Convert the first 'n' positional arguments after 'offset' if they are torch tensors
                end_index = min(offset + n, len(modified_args))
                for i in range(offset, end_index):
                    if isinstance(modified_args[i], torch.Tensor):
                        modified_args[i] = getter(modified_args[i]).astype(dtype)
            else:
                # Convert all torch tensor arguments after 'offset'
                for i in range(offset, len(modified_args)):
                    if isinstance(modified_args[i], torch.Tensor):
                        modified_args[i] = getter(modified_args[i]).astype(dtype)

            # Convert keyword arguments if they are specified in keys and are torch tensors
            if keys:
                for key in keys:
                    if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = getter(kwargs[key]).astype(dtype)

            return func(*modified_args, **kwargs)

        return wrapper

    return decorator
