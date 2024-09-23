from typing import Callable, Optional, Any, Dict
import numpy as np
import torch
from functools import wraps


def tensor_to_numpy(ts: torch.Tensor, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Convert a torch.Tensor to a numpy.ndarray.
    Ensures the tensor is detached and moved to CPU before conversion.
    """
    np_array = ts.detach().cpu().numpy()
    if dtype:
        np_array = np_array.astype(dtype)
    return np_array


def torch2np(dtype: Optional[np.dtype] = None, *keys: str, n: Optional[int] = None) -> Callable:
    """
    Convert torch tensors to numpy arrays.
    This decorator handles both class methods and standalone functions.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.
    If n is specified, only the first n positional arguments (after 'self' for methods) are converted.
    If 'n' is greater than the number of arguments, conversion proceeds without error.

    Parameters
    ----------
    dtype : np.dtype, optional
        The desired dtype for the numpy arrays. If None, keeps the default dtype of the tensors.
    *keys : str, optional
        The keys of the dictionary to convert. If not provided, positional arguments are used.
    n : int, optional
        The number of initial arguments to convert. If None, all arguments are converted.

    Returns
    -------
    Callable
        The decorated function with torch tensor arguments converted to numpy arrays.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if the function is a method (e.g., first arg is 'self')
            is_method = len(args) > 0 and hasattr(args[0], func.__name__)
            offset = 1 if is_method else 0

            modified_args = list(args)

            # If keys are provided, convert only dictionary keys
            if keys:
                target_dict = modified_args[offset] if offset < len(modified_args) else None
                if not isinstance(target_dict, dict):
                    raise TypeError(f"Expected a dictionary at position {offset} but got {type(target_dict)}")

                for key in keys:
                    if key not in target_dict:
                        raise KeyError(f"Key '{key}' not found in dictionary")
                    if isinstance(target_dict[key], torch.Tensor):
                        target_dict[key] = tensor_to_numpy(target_dict[key], dtype)

            # If 'n' is provided, convert the first 'n' positional arguments
            elif n is not None:
                for i in range(offset, min(offset + n, len(modified_args))):
                    if isinstance(modified_args[i], torch.Tensor):
                        modified_args[i] = tensor_to_numpy(modified_args[i], dtype)

            # Otherwise, convert all positional arguments that are tensors
            else:
                for i in range(offset, len(modified_args)):
                    if isinstance(modified_args[i], torch.Tensor):
                        modified_args[i] = tensor_to_numpy(modified_args[i], dtype)

            # Convert keyword arguments if they are specified in keys and are torch tensors
            for key in keys:
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = tensor_to_numpy(kwargs[key], dtype)

            return func(*modified_args, **kwargs)

        return wrapper

    return decorator
