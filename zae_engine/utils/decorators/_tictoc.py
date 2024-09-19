import time
from functools import wraps
from typing import Callable, Any


def tictoc(func: Callable) -> Callable:
    """
    Measure and print the elapsed time of a function or method execution.
    Automatically detects if it is used in a class method or a standalone function and behaves accordingly.

    Parameters
    ----------
    func : Callable
        The function or method to be timed.

    Returns
    -------
    Callable
        The decorated function or method with timing functionality.

    Examples
    --------
    >>> @tictoc
    ... def example_func():
    ...     time.sleep(1)
    >>> example_func()  # This will print the elapsed time.

    >>> class Example:
    ...     @tictoc
    ...     def example_method(self):
    ...         time.sleep(1)
    >>> example = Example()
    >>> example.example_method()  # This will print the elapsed time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Determine if this is a method by checking if the first argument is an instance with the function as an attribute
        if len(args) > 0 and hasattr(args[0], func.__name__):
            # Likely a method; 'self' is the first argument
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time [sec] (method '{func.__name__}'): {elapsed_time:.6f}")
        else:
            # Likely a standalone function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time [sec] (function '{func.__name__}'): {elapsed_time:.6f}")
        return result

    return wrapper
