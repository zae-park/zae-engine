from typing import Callable, Union, Any
from functools import wraps
import inspect

GITHUB_REPO = "https://github.com/zae-park/zae-engine/blob/main/zae_engine/utils/decorators/_shape_check.py"


def shape_check(*keys: Union[int, str]) -> Callable:
    """
    Ensure that the shapes of specified arguments are the same.
    This decorator automatically handles both class methods and standalone functions.
    """
    if len(keys) == 0:
        raise ValueError("At least one key or an integer specifying number of positional arguments must be provided.")

    # Validate keys
    if len(keys) == 1 and isinstance(keys[0], int):
        num_args = keys[0]
        if num_args <= 1:
            raise ValueError("Cannot compare shape of single argument or non-positive number.")
        check_by_name = False
    else:
        for k in keys:
            if not isinstance(k, str):
                raise ValueError(
                    "When multiple arguments are provided, they must be strings representing argument names."
                )
        check_by_name = True

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Use inspect to bind the arguments
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Exclude 'self' or 'cls' from arguments if present
                arguments = bound_args.arguments
                arg_values = []

                # Remove 'self' or 'cls' from arguments
                arg_items = list(arguments.items())
                if arg_items and arg_items[0][0] in ("self", "cls"):
                    arg_items = arg_items[1:]

                if check_by_name:
                    # keys are argument names
                    for key in keys:
                        if key in arguments:
                            arg = arguments[key]
                            arg_values.append(arg)
                        else:
                            raise KeyError(f'Argument "{key}" not found in function "{func.__name__}".')
                else:
                    # keys[0] is an integer specifying number of positional arguments
                    positional_args = []
                    for name, value in arg_items:
                        param = sig.parameters[name]
                        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                            positional_args.append(value)
                        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                            positional_args.extend(value)  # Unpack *args
                    if len(positional_args) < num_args:
                        raise AssertionError(
                            f"Expected at least {num_args} positional arguments (excluding 'self' or 'cls'), "
                            f"but got {len(positional_args)} in function '{func.__name__}'."
                        )
                    arg_values = positional_args[:num_args]

                # Now check the shapes
                shape_list = []
                for arg in arg_values:
                    if not hasattr(arg, "shape"):
                        raise TypeError(f"Argument ({arg}) does not have a 'shape' attribute.")
                    shape_list.append(arg.shape)
                # Check if all shapes are the same
                if not shape_list:
                    raise AssertionError("No shapes to compare.")
                first_shape = shape_list[0]
                for s in shape_list[1:]:
                    if s != first_shape:
                        raise AssertionError(f"Shapes of the given arguments are not the same: {shape_list}")
                # Proceed to call the function
                return func(*args, **kwargs)
            except Exception as e:
                error_message = (
                    f'An error occurred in function "{func.__name__}": {e}'
                    f"\nThis might be due to incorrect argument shapes or usage.\n"
                    f"For more information, visit: {GITHUB_REPO}"
                )
                # Re-raise the same exception type with new message
                raise type(e)(error_message).with_traceback(e.__traceback__)

        return wrapper

    return decorator
