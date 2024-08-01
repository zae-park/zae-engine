import time
from functools import wraps
from typing import Callable


def tictoc_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time [sec]: {elapsed_time}")
        return result

    return wrapper


def tictoc_mtd(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time [sec]: {elapsed_time}")
        return result

    return wrapper


def tictoc(func: Callable) -> Callable:
    """
    Measure and print the elapsed time of a function execution.
    This decorator automatically detects if it is used in a class method or a standalone function and behaves accordingly.

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

    >>> class Example:
    ...     @tictoc
    ...     def example_method(self):
    ...         time.sleep(1)
    >>> example = Example()
    >>> example.example_method()  # This will print the elapsed time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], func.__name__):
            # 메소드인 경우
            decorator = tictoc_mtd(func)
            return decorator(*args, **kwargs)
        else:
            # 함수인 경우
            decorator = tictoc_fn(func)
            return decorator(*args, **kwargs)

    return wrapper
