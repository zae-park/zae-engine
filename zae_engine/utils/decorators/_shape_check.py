from typing import Callable, Union, Any
from functools import wraps
import inspect

GITHUB_REPO = "https://github.com/zae-park/zae-engine/blob/main/zae_engine/utils/decorators/_shape_check.py"


def shape_check(*keys: Union[int, str]) -> Callable:
    """
    Ensure that the shapes of specified arguments are the same.
    This decorator automatically detects if it is used in a class method or a standalone function and behaves accordingly.

    Parameters
    ----------
    keys : int or str
        If a single integer is provided, it checks the shapes of the first 'keys' positional arguments.
        If multiple strings are provided, it checks the shapes of the corresponding keyword arguments.
    github_repo : str, optional
        URL to the GitHub repository containing the decorator's implementation and documentation.

    Returns
    -------
    Callable
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

    if len(keys) == 0:
        raise ValueError("At least one key or an integer specifying number of positional arguments must be provided.")

    # Validate keys
    if len(keys) == 1 and isinstance(keys[0], int):
        num_args = keys[0]
        if num_args <= 1:
            raise ValueError("Cannot compare shape of single argument or non-positive number.")
    else:
        for k in keys:
            if not isinstance(k, str):
                raise ValueError(
                    "When multiple arguments are provided, they must be strings representing keyword argument names."
                )

    def decorator(func: Callable) -> Callable:
        # Inspect the function signature to determine if it's a method
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        is_method = False
        offset = 0
        if params:
            first_param = params[0]
            if first_param.name in ("self", "cls"):
                is_method = True
                offset = 1

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                if len(keys) == 1 and isinstance(keys[0], int):
                    num_args = keys[0]
                    if len(args) < num_args + offset:
                        raise AssertionError(
                            f"Expected at least {num_args} positional arguments{' after self' if is_method else ''}, "
                            f"but got {len(args) - offset}."
                        )
                    # Extract shapes
                    shape_list = []
                    for i in range(offset, offset + num_args):
                        arg = args[i]
                        if not hasattr(arg, "shape"):
                            raise TypeError(f"Argument at position {i} ({arg}) does not have a 'shape' attribute.")
                        shape_list.append(arg.shape)
                else:
                    # keys are strings
                    shape_list = []
                    for key in keys:
                        if key not in kwargs:
                            raise KeyError(f'Keyword argument "{key}" not found.')
                        arg = kwargs[key]
                        if not hasattr(arg, "shape"):
                            raise TypeError(f"Keyword argument \"{key}\" ({arg}) does not have a 'shape' attribute.")
                        shape_list.append(arg.shape)

                # Check if all shapes are the same
                if not shape_list:
                    raise AssertionError("No shapes to compare.")
                first_shape = shape_list[0]
                for s in shape_list[1:]:
                    if s != first_shape:
                        raise AssertionError(f"Shapes of the given arguments are not the same: {shape_list}")

                return func(*args, **kwargs)
            except Exception as e:
                # Determine if it's a method or function based on is_method flag
                if is_method:
                    error_type = "method"
                else:
                    error_type = "function"
                error_message = (
                    f'An error occurred in ({error_type}) "{func.__name__}": {e}'
                    f"\nThis might be due to incorrect argument shapes or usage.\n"
                    f"For more information, visit: {GITHUB_REPO}"
                )
                # Re-raise the same exception type with new message
                raise type(e)(error_message).with_traceback(e.__traceback__)

        return wrapper

    return decorator
