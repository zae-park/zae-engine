from typing import Callable, Optional, Any, Dict
import numpy as np
import torch
from functools import wraps


def np2torch(dtype: torch.dtype, *keys: str, n: Optional[int] = None) -> Callable:
    """
    Convert numpy arrays to torch tensors with a specified dtype.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.
    If a key is missing, a KeyError is raised.
    """

    def convert_to_tensor(array: np.ndarray) -> torch.Tensor:
        """Helper function to convert numpy array to torch tensor."""
        try:
            return torch.from_numpy(array).type(dtype)
        except TypeError:
            return torch.tensor(array, dtype=dtype)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Determine if this is a method by checking if the first argument has the function as an attribute
            is_method = len(args) > 0 and hasattr(args[0], func.__name__)
            offset = 1 if is_method else 0

            modified_args = list(args)

            # Handle conversion of dictionary values if keys are specified
            if keys:
                if offset >= len(modified_args):
                    raise ValueError(f"Expected a dictionary at position {offset}, but got fewer arguments.")
                target_arg = modified_args[offset]
                if not isinstance(target_arg, dict):
                    raise TypeError(f"Expected a dictionary at position {offset}, but got {type(target_arg)}.")
                for key in keys:
                    if key not in target_arg:
                        raise KeyError(f"Key '{key}' not found in the input dictionary.")
                    if isinstance(target_arg[key], np.ndarray):
                        target_arg[key] = convert_to_tensor(target_arg[key])

            elif n is not None:
                # Convert the first 'n' positional arguments after 'offset' if they are numpy arrays
                end_index = min(offset + n, len(modified_args))
                for i in range(offset, end_index):
                    if isinstance(modified_args[i], np.ndarray):
                        modified_args[i] = convert_to_tensor(modified_args[i])

            # Convert all numpy arrays in remaining positional arguments if n is not specified
            else:
                for i in range(offset, len(modified_args)):
                    if isinstance(modified_args[i], np.ndarray):
                        modified_args[i] = convert_to_tensor(modified_args[i])

            # Convert specified keyword arguments (if keys are given)
            for key in keys:
                if key in kwargs and isinstance(kwargs[key], np.ndarray):
                    kwargs[key] = convert_to_tensor(kwargs[key])

            return func(*modified_args, **kwargs)

        return wrapper

    return decorator
