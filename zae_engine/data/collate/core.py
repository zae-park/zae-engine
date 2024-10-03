import copy
import logging
from functools import wraps
from collections import defaultdict, OrderedDict
from typing import Union, Callable, List, Tuple, Dict, Iterator, Sequence, Any

import torch

logger = logging.getLogger(__name__)


class CollateBase:
    """
    Base class for collating and processing batches of data using a sequence of functions.

    This class allows you to define a sequence of preprocessing functions that will be applied
    to data batches in the specified order. It supports initialization with either an OrderedDict
    or a list of functions.

    Parameters
    ----------
    x_key : Sequence[str], default=["x"]
        The key(s) in the batch dictionary that represent the input data.
    y_key : Sequence[str], default=["y"]
        The key(s) in the batch dictionary that represent the labels.
    aux_key : Sequence[str], default=["aux"]
        The key(s) in the batch dictionary that represent the auxiliary data.
    functions : Union[List[Callable], OrderedDict[str, Callable]], optional
        The preprocessing functions to apply to the batches in sequence.

    Methods
    -------
    __len__():
        Returns the number of functions in the collator.
    __iter__():
        Returns an iterator over the functions in the collator.
    io_check(sample_data: Union[dict, OrderedDict], check_all: bool = False) -> None:
        Checks if the registered functions maintain the structure of the sample data.
    set_batch(batch: Union[dict, OrderedDict]) -> None:
        Sets the sample batch for structure checking.
    add_fn(name: str, fn: Callable) -> None:
        Adds a function to the collator with the given name.
    __call__(batch: List[Union[dict, OrderedDict]]) -> Union[dict, OrderedDict]:
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
    >>> collator = CollateBase(x_key=['x'], y_key=['y'], aux_key=['aux'], functions=[fn1, fn2])
    >>> batch = {'x': [1, 2, 3], 'y': [1], 'aux': [0.5], 'filename': 'sample.txt'}
    >>> processed_batch = collator([batch, batch])

    Example 2: Initialization with an OrderedDict
    >>> from collections import OrderedDict
    >>> functions = OrderedDict([('fn1', fn1), ('fn2', fn2)])
    >>> collator = CollateBase(x_key=['x'], y_key=['y'], aux_key=['aux'], functions=functions)
    >>> processed_batch = collator([batch, batch])

    Example 3: Checking input-output consistency
    >>> sample_data = {'x': [1, 2, 3], 'y': [1], 'aux': [0.5], 'filename': 'sample.txt'}
    >>> collator.set_batch(sample_data)
    >>> collator.add_fn('fn3', fn3)  # This will check if fn3 maintains the structure of sample_data
    """

    _fn: OrderedDict[str, Callable]

    def __init__(
        self,
        *,
        x_key: Sequence[str] = ("x",),
        y_key: Sequence[str] = ("y",),
        aux_key: Sequence[str] = ("aux",),
        functions: Union[List[Callable], OrderedDict[str, Callable]] = None,
    ):
        self.x_key, self.y_key, self.aux_key = x_key, y_key, aux_key
        self.sample_batch = {}
        self._fn = OrderedDict()  # Initialize the ordered dictionary
        self._fn_checked = OrderedDict()  # Track which functions have passed io_check

        if functions:
            if isinstance(functions, OrderedDict):
                for key, module in functions.items():
                    self.add_fn(key, module)
            elif isinstance(functions, list):
                for idx, module in enumerate(functions):
                    self.add_fn(str(idx), module)
            else:
                raise TypeError("functions must be a list or an OrderedDict of callables.")

    def __len__(self) -> int:
        return len(self._fn)

    def __iter__(self) -> Iterator:
        return iter(self._fn.values())

    def io_check(self, sample_data: Union[dict, OrderedDict], check_all: bool = False) -> None:
        """
        Checks if the registered functions maintain the structure of the sample batch.
        Only checks the newly added function if check_all is False.

        Parameters
        ----------
        sample_data : Union[dict, OrderedDict]
            The sample data to test the functions with.
        check_all : bool
            If True, checks all registered functions. Otherwise, checks only the newly added function.
        """

        def check(updated_sample_data: Union[dict, OrderedDict], fn_list):
            """
            Internal function to check the sample data against the provided functions.

            Parameters
            ----------
            updated_sample_data : Union[dict, OrderedDict]
                The sample data to be processed.
            fn_list : list
                List of functions to apply to the sample data.
            """
            if not updated_sample_data:
                raise ValueError("Sample data cannot be empty for io_check.")

            # Make a deep copy to avoid modifying the original sample_data
            updated_sample_data = copy.deepcopy(updated_sample_data)

            # Ensure the function list is iterated correctly
            for fn_name, fn in fn_list:
                updated_sample_data = fn(updated_sample_data)
                # Mark function as checked
                self._fn_checked[fn_name] = True

            # Check structure integrity
            assert isinstance(updated_sample_data, type(sample_data)), "The functions changed the type of the batch."
            assert set(sample_data.keys()).issubset(
                updated_sample_data.keys()
            ), "The functions changed the keys of the batch."

            for key in sample_data.keys():
                assert isinstance(
                    updated_sample_data[key], type(sample_data[key])
                ), f"The type of value for key '{key}' has changed."
                if not isinstance(updated_sample_data[key], (list, str)):
                    assert (
                        updated_sample_data[key].dtype == sample_data[key].dtype
                    ), f"The dtype of value for key '{key}' has changed."

        # Choose functions to check
        if check_all:
            functions_to_check = [(name, fn) for name, fn in self._fn.items()]
        else:
            # Only check functions that haven't been checked yet
            functions_to_check = [(name, fn) for name, fn in self._fn.items() if not self._fn_checked.get(name, False)]

        # Perform the check
        check(sample_data, functions_to_check)

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
        if name in self._fn:
            raise ValueError(f"Function with name '{name}' already exists.")
        self._fn[name] = fn
        self._fn_checked[name] = False  # Mark as unchecked

        if self.sample_batch:
            # Check only the newly added function
            self.io_check(self.sample_batch)

    def set_batch(self, batch: Union[dict, OrderedDict]) -> None:
        """
        Sets the sample batch to be used for input-output structure validation.

        Parameters
        ----------
        batch : Union[dict, OrderedDict]
            The sample batch.
        """
        self.sample_batch = batch
        # Reset all functions to unchecked
        for key in self._fn_checked:
            self._fn_checked[key] = False

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

        accumulate_dict = defaultdict(list)
        for idx, b in enumerate(batches):
            # Check for missing keys
            for key in self.x_key + self.y_key + self.aux_key:
                if key not in b:
                    raise KeyError(f"Batch at index {idx} is missing required key: '{key}'")
            for k, v in b.items():
                accumulate_dict[k].append(v)

        for k, v in accumulate_dict.items():
            try:
                if k in self.x_key:
                    accumulate_dict[k] = torch.stack(v, dim=0) if v else []
                elif k in self.y_key:
                    accumulate_dict[k] = torch.stack(v, dim=0).squeeze()
                else:
                    accumulate_dict[k] = v
            except (TypeError, RuntimeError) as e:
                logger.error(f"Error accumulating key '{k}': {e}")
                raise type(e)(f"Error accumulating key '{k}': {e}") from e  # Raise with additional context

        return accumulate_dict

    def __call__(self, batch: List[Union[dict, OrderedDict]]) -> Union[dict, OrderedDict]:
        """
        Applies the preprocessing functions to the input batch in sequence.

        Parameters
        ----------
        batch : List[Union[dict, OrderedDict]]
            The input batch to be processed.

        Returns
        -------
        Union[dict, OrderedDict]
            The processed and accumulated batch.
        """
        processed_batches = []
        for b in copy.deepcopy(batch):
            for fn in self._fn.values():
                b = fn(b)
            processed_batches.append(b)
        accumulated = self.accumulate(processed_batches)
        return accumulated

    def wrap(self, func: Callable = None):
        if func is None:
            func = self.__call__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func
