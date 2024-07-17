from typing import Union, Callable, List, Tuple, OrderedDict, overload, Iterator
from abc import ABC, abstractmethod


class CollateBase(ABC):
    """
    Base class for collating and processing batches of data using a sequence of functions.

    This class allows you to define a sequence of preprocessing functions that will be applied
    to data batches in the specified order. It supports initialization with either an OrderedDict
    or a list of functions.

    Methods
    -------
    __len__():
        Returns the number of functions in the collator.
    __iter__():
        Returns an iterator over the functions in the collator.
    io_check(sample_data: Union[dict, OrderedDict]) -> None:
        Checks if the registered functions maintain the structure of the sample data.
    set_batch(batch: Union[dict, OrderedDict]) -> None:
        Sets the sample batch for structure checking.
    add_fn(name: str, fn: Callable) -> None:
        Adds a function to the collator with the given name.
    __call__(batch: Union[dict, OrderedDict]) -> Union[dict, OrderedDict]:
        Applies the registered functions to the input batch in sequence.

    Usage
    -----
    Example 1: Initialization with a list of functions
    >>> def fn1(batch):
    >>>     # Function to process batch
    >>>     return batch
    >>> def fn2(batch):
    >>>     # Another function to process batch
    >>>     return batch
    >>> collator = CollateBase(fn1, fn2)
    >>> batch = {'data': [1, 2, 3]}
    >>> processed_batch = collator(batch)

    Example 2: Initialization with an OrderedDict
    >>> from collections import OrderedDict
    >>> functions = OrderedDict([('fn1', fn1), ('fn2', fn2)])
    >>> collator = CollateBase(functions)
    >>> processed_batch = collator(batch)

    Example 3: Checking input-output consistency
    >>> sample_data = {'data': [1, 2, 3]}
    >>> collator.io_check(sample_data)
    >>> collator.set_batch(sample_data)
    >>> collator.add_fn('fn3', fn3)  # This will check if fn3 maintains the structure of sample_data
    """

    _fn: OrderedDict[str, Callable]

    @overload
    def __init__(self, *args: Callable) -> None: ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Callable]") -> None: ...

    def __init__(self, *args):
        super().__init__()
        self._fn = OrderedDict()  # Initialize the ordered dictionary
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_fn(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_fn(str(idx), module)
        self.sample_batch = {}

    def __len__(self) -> int:
        return len(self._fn)

    def __iter__(self) -> Iterator:
        return iter(self._fn.values())

    def io_check(self, sample_data: Union[dict, OrderedDict]) -> None:
        """
        Checks if the registered functions maintain the structure of the sample batch.

        Parameters
        ----------
        sample_data : Union[dict, OrderedDict]
            The sample data to test the functions with.

        Raises
        ------
        AssertionError
            If any function changes the structure of the sample data.
        """
        if not sample_data:
            raise ValueError("Sample data cannot be empty for io_check.")

        self.set_batch(sample_data)  # Update the sample_batch with the provided sample_data
        keys = self.sample_batch.keys()
        updated = self.sample_batch
        for fn in self._fn.values():
            updated = fn(updated)
        assert isinstance(updated, type(self.sample_batch)), "The functions changed the type of the batch."
        assert set(keys).issubset(updated.keys()), "The functions changed the keys of the batch."

    def set_batch(self, batch: Union[dict, OrderedDict]) -> None:
        """
        Sets the sample batch to be used for input-output structure validation.

        Parameters
        ----------
        batch : Union[dict, OrderedDict]
            The sample batch.
        """
        self.sample_batch = batch

    def add_fn(self, name: str, fn: Callable) -> None:
        """
        Adds a new preprocessing function to the pipeline after validation.

        Parameters
        ----------
        name : str
            The name of the function.
        fn : Callable
            The preprocessing function.
        """
        self.io_check(self.sample_batch)
        self._fn[name] = fn

    def __call__(self, batch: Union[dict, OrderedDict]) -> Union[dict, OrderedDict]:
        """
        Applies the preprocessing functions to the input batch in order.

        Parameters
        ----------
        batch : Union[dict, OrderedDict]
            The input batch to be processed.

        Returns
        -------
        Union[dict, OrderedDict]
            The processed batch.
        """
        for fn in self._fn.values():
            batch = fn(batch)
        return batch

    def wrap(self, func: Callable = None):
        if func is None:
            func = self.__call__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func
