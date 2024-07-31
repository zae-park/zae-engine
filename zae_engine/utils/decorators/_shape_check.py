from typing import Callable, Union


def shape_check_fn(*keys: Union[int, str]) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if isinstance(keys[0], int):
                num_args = keys[0]
                shape_list = [a.shape for a in args[:num_args]]
            else:
                shape_list = [kwargs[key].shape for key in keys if key in kwargs]
            if len(set(shape_list)) != 1:
                raise AssertionError("Shapes of the given arguments are not the same.")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def shape_check_mtd(*keys: Union[int, str]) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if isinstance(keys[0], int):
                num_args = keys[0]
                shape_list = [a.shape for a in args[:num_args]]
            else:
                shape_list = [kwargs[key].shape for key in keys if key in kwargs]
            if len(set(shape_list)) != 1:
                raise AssertionError("Shapes of the given arguments are not the same.")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def shape_check(*keys: Union[int, str]) -> Callable:
    """
    Ensure that the shapes of specified arguments are the same.
    This decorator automatically detects if it is used in a class method or a standalone function and behaves accordingly.

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
            if len(args) > 0 and hasattr(args[0], func.__name__):
                # 메소드인 경우
                decorator = shape_check_mtd(*keys)
                return decorator(func)(*args, **kwargs)
            else:
                # 함수인 경우
                decorator = shape_check_fn(*keys)
                return decorator(func)(*args, **kwargs)

        return wrapper

    return deco
