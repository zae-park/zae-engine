from collections import defaultdict
from typing import Union, Callable, List, Tuple, Dict, OrderedDict, overload, Iterator, Sequence
from functools import wraps
from abc import ABC, abstractmethod

import torch
from zae_engine.utils.decorators import np2torch


class CollateBase(ABC):
    """
    Base class for collating and processing batches of data using a sequence of functions.

    This class allows you to define a sequence of preprocessing functions that will be applied
    to data batches in the specified order. It supports initialization with either an OrderedDict
    or a list of functions.

    Parameters
    ----------
    x_key : List[str]
        The key in the batch dictionary that represents the input data.
    y_key : List[str]
        The key in the batch dictionary that represents the labels.
    aux_key : List[str]
        The key in the batch dictionary that represents the auxiliary data.
    args : Union[OrderedDict[str, Callable], List[Callable]]
        The functions to be applied to the batches in sequence.

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
    accumulate(batches: Union[Tuple, List]) -> Dict:
        Convert a list of dictionaries per data to a batch dictionary with list-type values.

    Usage
    -----
    Example 1: Initialization with a list of functions
    >>> def fn1(batch):
    >>>     # Function to process batch
    >>>     return batch
    >>> def fn2(batch):
    >>>     # Another function to process batch
    >>>     return batch
    >>> collator = CollateBase(x_key=['x'], y_key=['y'], aux_key=['aux'], fn1, fn2)
    >>> batch = {'data': [1, 2, 3], 'label': [1], 'aux': [0.5], 'filename': 'sample.txt'}
    >>> processed_batch = collator([batch, batch])

    Example 2: Initialization with an OrderedDict
    >>> from collections import OrderedDict
    >>> functions = OrderedDict([('fn1', fn1), ('fn2', fn2)])
    >>> collator = CollateBase(x_key=['x'], y_key=['y'], aux_key=['aux'], functions)
    >>> processed_batch = collator([batch, batch])

    Example 3: Checking input-output consistency
    >>> sample_data = {'x': [1, 2, 3], 'y': [1], 'aux': [0.5], 'filename': 'sample.txt'}
    >>> collator.io_check(sample_data)
    >>> collator.set_batch(sample_data)
    >>> collator.add_fn('fn3', fn3)  # This will check if fn3 maintains the structure of sample_data
    """

    _fn: OrderedDict[str, Callable]

    @overload
    def __init__(self, x_key: Sequence[str], y_key: Sequence[str], aux_key: Sequence[str], *args: Callable) -> None: ...

    @overload
    def __init__(
        self, x_key: Sequence[str], y_key: Sequence[str], aux_key: Sequence[str], arg: "OrderedDict[str, Callable]"
    ) -> None: ...

    def __init__(
        self, *args, x_key: Sequence[str] = ["x"], y_key: Sequence[str] = ["y"], aux_key: Sequence[str] = ["aux"]
    ):
        super().__init__()
        self.x_key, self.y_key, self.aux_key = x_key, y_key, aux_key
        self.sample_batch = {}
        self._fn = OrderedDict()  # Initialize the ordered dictionary
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_fn(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_fn(str(idx), module)

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

        @np2torch(torch.float, *(self.x_key + self.y_key))
        def check(sample_data_inner: Union[dict, OrderedDict]):
            if not sample_data_inner:
                raise ValueError("Sample data cannot be empty for io_check.")

            self.set_batch(sample_data_inner)  # Update the sample_batch with the provided sample_data
            keys = self.sample_batch.keys()
            updated = self.sample_batch.copy()
            for fn in self._fn.values():
                updated = fn(updated)
            assert isinstance(updated, type(self.sample_batch)), "The functions changed the type of the batch."
            assert set(keys).issubset(updated.keys()), "The functions changed the keys of the batch."

            for key in keys:
                assert isinstance(
                    updated[key], type(self.sample_batch[key])
                ), f"The type of value for key '{key}' has changed."

                if not isinstance(updated[key], (list, str)):
                    assert (
                        updated[key].dtype == self.sample_batch[key].dtype
                    ), f"The dtype of value for key '{key}' has changed."

        check(sample_data)

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
        self._fn[name] = fn
        if self.sample_batch:
            self.io_check(self.sample_batch)

    def accumulate(self, batches: Union[Tuple, List]) -> Dict:
        """
        Convert a list of dictionaries per data to a batch dictionary with list-type values.

        Parameters
        ----------
        batches : Union[Tuple, List]
            A list of dictionaries, where each dictionary represents a data batch.

        Returns
        -------
        Dict
            A dictionary where keys are batch attributes and values are lists or concatenated tensors.
        """

        @np2torch(torch.float, *(self.x_key + self.y_key))
        def torch_sweep(**kwargs):
            return kwargs

        accumulate_dict = defaultdict(list)
        for b in batches:
            for k, v in torch_sweep(**b).items():
                accumulate_dict[k].append(v)
        for k, v in accumulate_dict.items():
            try:
                if k in self.x_key:
                    accumulate_dict[k] = torch.stack(v, dim=0).unsqueeze(1) if v else []
                elif k in self.y_key:
                    accumulate_dict[k] = torch.stack(v, dim=0).squeeze()
                else:
                    accumulate_dict[k] = v
            except TypeError as e:
                print(e)
                pass
        return accumulate_dict

    def __call__(self, batch: List[Union[dict, OrderedDict]]) -> Union[dict, OrderedDict]:
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
        batches = []
        for b in batch:
            for fn in self._fn.values():
                b = fn(b)
            batches.append(b)
        batches = self.accumulate(batches)
        return batches

    def wrap(self, func: Callable = None):
        if func is None:
            func = self.__call__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func
