import time
import inspect

from typing import Callable, Union
import numpy as np
import torch


def np2torch(dtype: torch.dtype, *keys: str, n: int = None) -> Callable:
    """
    Convert numpy arrays to torch tensors with a specified dtype.

    This decorator converts specified numpy array arguments of a function to torch tensors with the specified dtype.
    If an argument is already a torch tensor, it is not converted. If n is not specified, all numpy array arguments are converted.
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

    >>> @np2torch(torch.float32, "x")
    ... def example_func(batch):
    ...     return batch
    >>> example_func({'x': np.array([1, 2, 3]), 'y': [4, 5, 6]})
    # This will convert only the value of 'x' key in the dictionary to torch tensor.
    """

    def deco(func: Callable) -> Callable:
        def wrapper(*args: Union[np.ndarray, torch.Tensor, bool, int, float], **kwargs):
            instance_args = args[1:] if len(args) > 0 and hasattr(args[0], "__class__") else args
            instance = len(args) > 0 and hasattr(args[0], "__class__")
            if keys:
                if len(instance_args) > 0 and isinstance(instance_args[0], dict):
                    instance_args = (
                        dict(
                            instance_args[0],
                            **{
                                k: torch.tensor(instance_args[0][k], dtype=dtype) for k in keys if k in instance_args[0]
                            },
                        ),
                    ) + instance_args[1:]
                elif isinstance(kwargs, dict):
                    kwargs = {**kwargs, **{k: torch.tensor(kwargs[k], dtype=dtype) for k in keys if k in kwargs}}
            else:
                if n is None:
                    n_args = len(instance_args)
                else:
                    n_args = min(n, len(instance_args))

                instance_args = tuple(
                    torch.tensor(a, dtype=dtype) if isinstance(a, np.ndarray) and i < n_args else a
                    for i, a in enumerate(instance_args)
                )

                kwargs = {
                    k: torch.tensor(v, dtype=dtype) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
                }

            if instance:
                return func(args[0], *instance_args, **kwargs)
            else:
                return func(*instance_args, **kwargs)

        return wrapper

    return deco


def torch2np(dtype: np.dtype, *keys: str, n: int = None) -> Callable:
    """
    Convert torch tensors to numpy arrays with a specified dtype.

    This decorator converts specified torch tensor arguments of a function to numpy arrays with the specified dtype.
    If an argument is already a numpy array, it is not converted. If n is not specified, all torch tensor arguments are converted.
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

    >>> @torch2np(np.float32, key='x')
    ... def example_func(batch):
    ...     return batch
    >>> example_func({'x': torch.tensor([1, 2, 3]), 'y': [4, 5, 6]})
    # This will convert only the value of 'x' key in the dictionary to numpy array.
    """

    def deco(func: Callable) -> Callable:
        def wrapper(*args: Union[np.ndarray, torch.Tensor, bool, int, float], **kwargs):
            instance_args = args[1:] if len(args) > 0 and hasattr(args[0], "__class__") else args
            instance = len(args) > 0 and hasattr(args[0], "__class__")
            if keys:
                if len(instance_args) > 0 and isinstance(instance_args[0], dict):
                    instance_args = (
                        dict(
                            instance_args[0],
                            **{
                                k: instance_args[0][k].clone().detach().numpy().astype(dtype)
                                for k in keys
                                if k in instance_args[0]
                            },
                        ),
                    ) + instance_args[1:]
                elif isinstance(kwargs, dict):
                    kwargs = {
                        **kwargs,
                        **{k: kwargs[k].clone().detach().numpy().astype(dtype) for k in keys if k in kwargs},
                    }
            else:
                if n is None:
                    n_args = len(instance_args)
                else:
                    n_args = min(n, len(instance_args))

                instance_args = tuple(
                    a.clone().detach().numpy().astype(dtype) if isinstance(a, torch.Tensor) and i < n_args else a
                    for i, a in enumerate(instance_args)
                )

                kwargs = {
                    k: v.clone().detach().numpy().astype(dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }

            if instance:
                return func(args[0], *instance_args, **kwargs)
            else:
                return func(*instance_args, **kwargs)

        return wrapper

    return deco


def shape_check(*keys):
    """
    Ensure that the shapes of specified arguments are the same.

    This decorator checks that the shapes of the specified arguments (either positional or keyword arguments) are the same.
    If the shapes do not match, an AssertionError is raised.

    Parameters
    ----------
    keys : int or str
        If a single integer is provided, it checks the shapes of the first 'keys' positional arguments.
        If multiple strings are provided, it checks the shapes of the corresponding keyword arguments.

    Returns
    -------
    func
        The decorated function with shape checks on the specified arguments.

    Examples
    --------
    >>> @shape_check(2)
    ... def example_func(x, y):
    ...     return x + y
    >>> @shape_check('x', 'y')
    ... def example_func(**kwargs):
    ...     return kwargs['x'] + kwargs['y']
    """
    if len(keys) == 1 and isinstance(keys[0], int):
        num_args = keys[0]
        if num_args <= 1:
            raise AssertionError("Cannot compare shape of single argument")
    else:
        for k in keys:
            if not isinstance(k, str):
                raise AssertionError("Input the multiple arg strings to check")

    def deco(func):
        def wrapper(*args, **kwargs):
            if isinstance(keys[0], int):
                shape_list = [a.shape for a in args[:num_args]]
            else:
                shape_list = [kwargs[key].shape for key in keys if key in kwargs]
            if len(set(shape_list)) != 1:
                raise AssertionError("Shape of given args is not same.")
            return func(*args, **kwargs)

        return wrapper

    return deco


def tictoc(func):
    """
    Measure and print the elapsed time of a function execution.

    This decorator measures the time it takes to execute the decorated function and prints the elapsed time in seconds.

    Parameters
    ----------
    func : function
        The function to be timed.

    Returns
    -------
    func
        The decorated function with timing functionality.

    Examples
    --------
    >>> @tictoc
    ... def example_func():
    ...     time.sleep(1)
    >>> example_func()  # This will print the elapsed time.
    """

    def wrapper(*args, **kwargs):
        import time

        kickoff = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - kickoff
        print(f"Elapsed time [sec]: {elapsed_time}")
        return out

    return wrapper
