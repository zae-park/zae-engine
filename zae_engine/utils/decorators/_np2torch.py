from typing import Callable, Optional, Any, Dict
import numpy as np
import torch
from functools import wraps


def np2torch(
    dtype: torch.dtype, *keys: str, n: Optional[int] = None, device: Optional[torch.device] = None
) -> Callable:
    """
    Convert numpy arrays to torch tensors with a specified dtype and device.
    This decorator handles both class methods and standalone functions.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.
    If n is specified, only the first n positional arguments (after 'self' for methods) are converted.
    If 'n' is greater than the number of arguments, conversion proceeds without error.

    Parameters
    ----------
    dtype : torch.dtype
        The desired dtype for the torch tensors.
    *keys : str, optional
        The keys of the dictionary to convert. If not provided, arguments are converted based on 'n'.
    n : int, optional
        The number of initial arguments to convert. If None, all numpy array arguments are converted.
        If 'n' is greater than the number of arguments, conversion proceeds without error.
    device : torch.device, optional
        The device to place the torch tensors on. If None, tensors are created on the CPU.

    Returns
    -------
    Callable
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
                    if isinstance(v, np.ndarray):
                        try:
                            # Attempt to use torch.from_numpy for efficiency
                            tensor = torch.from_numpy(v).type(dtype)
                            if device is not None:
                                tensor = tensor.to(device)
                            target_arg[key] = tensor
                        except Exception:
                            # Fallback to torch.tensor if from_numpy fails
                            target_arg[key] = torch.tensor(v, dtype=dtype, device=device)
            elif n is not None:
                # Convert the first 'n' positional arguments after 'offset' if they are numpy arrays
                end_index = min(offset + n, len(modified_args))
                for i in range(offset, end_index):
                    if isinstance(modified_args[i], np.ndarray):
                        try:
                            tensor = torch.from_numpy(modified_args[i]).type(dtype)
                            if device is not None:
                                tensor = tensor.to(device)
                            modified_args[i] = tensor
                        except Exception:
                            modified_args[i] = torch.tensor(modified_args[i], dtype=dtype, device=device)
            else:
                # Convert all numpy array arguments after 'offset'
                for i in range(offset, len(modified_args)):
                    if isinstance(modified_args[i], np.ndarray):
                        try:
                            tensor = torch.from_numpy(modified_args[i]).type(dtype)
                            if device is not None:
                                tensor = tensor.to(device)
                            modified_args[i] = tensor
                        except Exception:
                            modified_args[i] = torch.tensor(modified_args[i], dtype=dtype, device=device)

            # Convert keyword arguments if they are specified in keys and are numpy arrays
            if keys:
                for key in keys:
                    if key in kwargs and isinstance(kwargs[key], np.ndarray):
                        try:
                            tensor = torch.from_numpy(kwargs[key]).type(dtype)
                            if device is not None:
                                tensor = tensor.to(device)
                            kwargs[key] = tensor
                        except Exception:
                            kwargs[key] = torch.tensor(kwargs[key], dtype=dtype, device=device)

            return func(*modified_args, **kwargs)

        return wrapper

    return decorator
