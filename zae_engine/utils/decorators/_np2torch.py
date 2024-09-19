# import inspect
# from typing import Union, Callable
# import numpy as np
# import torch
#
#
# def np2torch_fn(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
#     def decorator(func: Callable) -> Callable:
#         def wrapper(*args, **kwargs):
#             new_args = list(args)
#             if keys:
#                 for key in keys:
#                     if key in new_args[0]:
#                         new_args[0][key] = torch.tensor(new_args[0][key], dtype=dtype)
#             else:
#                 for i in range(n or len(new_args)):
#                     if isinstance(new_args[i], np.ndarray):
#                         new_args[i] = torch.tensor(new_args[i], dtype=dtype)
#             return func(*new_args, **kwargs)
#
#         return wrapper
#
#     return decorator
#
#
# def np2torch_mtd(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
#     def decorator(func: Callable) -> Callable:
#         def wrapper(self, *args, **kwargs):
#             new_args = list(args)
#             if keys:
#                 for key in keys:
#                     if key in new_args[0]:
#                         new_args[0][key] = torch.tensor(new_args[0][key], dtype=dtype)
#             else:
#                 for i in range(n or len(new_args)):
#                     if isinstance(new_args[i], np.ndarray):
#                         new_args[i] = torch.tensor(new_args[i], dtype=dtype)
#             return func(self, *new_args, **kwargs)
#
#         return wrapper
#
#     return decorator
#
#
# def np2torch(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
#     """
#     Convert numpy arrays to torch tensors with a specified dtype.
#     This decorator automatically detects if it is used in a class method or a standalone function and behaves accordingly.
#     If keys are specified, only the values corresponding to the keys in the dictionary are converted.
#
#     Parameters
#     ----------
#     dtype : torch.dtype
#         The desired dtype for the torch tensors.
#     n : int, optional
#         The number of initial arguments to convert. If None, all numpy array arguments are converted.
#     keys : str, optional
#         The keys of the dictionary to convert. If None, all arguments are converted.
#
#     Returns
#     -------
#     func
#         The decorated function with numpy array arguments converted to torch tensors.
#
#     Examples
#     --------
#     >>> @np2torch(torch.float32, n=2)
#     ... def example_func(x, y, z):
#     ...     return x, y, z
#     >>> example_func(np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]))
#     # This will convert only the first two numpy arrays to torch tensors.
#
#     >>> class Example:
#     ...     @np2torch(torch.float32, "x", "y")
#     ...     def example_method(self, batch):
#     ...         return batch
#     >>> example = Example()
#     >>> example.example_method({"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])})
#     # This will convert the values of 'x' and 'y' keys in the dictionary to torch tensors.
#     """
#
#     def deco(func: Callable) -> Callable:
#         def wrapper(*args, **kwargs):
#             if len(args) > 0 and hasattr(args[0], func.__name__):
#                 # 메소드인 경우
#                 decorator = np2torch_mtd(dtype, *keys, n=n)
#                 return decorator(func)(*args, **kwargs)
#             else:
#                 # 함수인 경우
#                 decorator = np2torch_fn(dtype, *keys, n=n)
#                 return decorator(func)(*args, **kwargs)
#
#         return wrapper
#
#     return deco
#
from typing import Callable
import numpy as np
import torch


def np2torch(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
    """
    Convert numpy arrays to torch tensors with a specified dtype.
    This decorator handles both class methods and standalone functions.
    If keys are specified, only the values corresponding to the keys in the dictionary are converted.
    If n is specified, only the first n arguments (after 'self' for methods) are converted.
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
                    # Directly access the key without checking existence to raise KeyError if missing
                    v = new_args[offset][key]
                    if isinstance(v, np.ndarray):
                        new_args[offset][key] = torch.tensor(v, dtype=dtype)
            else:
                # Adjust end_index to avoid IndexError if n is greater than the number of arguments
                end_index = min(offset + n, len(new_args)) if n is not None else len(new_args)
                for i in range(offset, end_index):
                    if isinstance(new_args[i], np.ndarray):
                        new_args[i] = torch.tensor(new_args[i], dtype=dtype)

            return func(*new_args, **kwargs)

        return wrapper

    return decorator
