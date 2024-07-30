import time

from typing import Union, Type, Callable
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
        def wrapper(fn: Callable, *args):
            if keys:
                assert (len(args) == 1) and isinstance(args[0], dict)
                args = [
                    {
                        k: torch.tensor(v, dtype=dtype) if (k in keys) and isinstance(v, np.ndarray) else v
                        for k, v in args[0].items()
                    }
                ]
            else:
                n_args = len(args) if n is None else min(n, len(args))

                args = (
                    torch.tensor(a, dtype=dtype) if isinstance(a, np.ndarray) and i < n_args else a
                    for i, a in enumerate(args)
                )

            return func(fn, *args)

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
        def wrapper(fn: Callable, *args):
            if keys:
                assert (len(args) == 1) and isinstance(args[0], dict)
                args = (
                    {
                        k: (
                            v.clone().detach().numpy().astype(dtype)
                            if (k in keys) and isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in args[0].items()
                    },
                )
            else:
                n_args = len(args) if n is None else min(n, len(args))

                args = tuple(
                    a.clone().detach().numpy().astype(dtype) if isinstance(a, torch.Tensor) and i < n_args else a
                    for i, a in enumerate(args)
                )

            return func(fn, *args)

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
