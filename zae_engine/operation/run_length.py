from dataclasses import dataclass
from typing import List, Union, Optional
from itertools import groupby


@dataclass
class Run:
    """
    Represents a run in Run-Length Encoding (RLE).

    Attributes
    ----------
    start_index : int
        The starting index of the run.
    end_index : int
        The ending index of the run.
    value : int
        The value of the run.
    """

    start_index: int
    end_index: int
    value: int

    def __repr__(self):
        """Provide a string representation of the Run object."""
        return f"Run(start_index={self.start_index}, end_index={self.end_index}, value={self.value})"


class RunList:
    """
    Stores the results of Run-Length Encoding (RLE) and provides methods
    to access raw and filtered runs.

    Attributes
    ----------
    all_runs : List[Run]
        A list of all runs without any filtering.
    sense : int
        The minimum length of runs to be considered in filtering.
    original_length : int
        The length of the original list that was encoded.
    """

    def __init__(self, all_runs: List[Run], sense: int, original_length: int):
        """
        Initializes the RunList object.

        Parameters
        ----------
        all_runs : List[Run]
            A list of all runs obtained from encoding.
        sense : int
            The minimum length of runs to be considered in filtering.
        original_length : int
            The length of the original list that was encoded.
        """
        self.all_runs = all_runs
        self.sense = sense
        self.original_length = original_length

    def raw(self) -> List[Run]:
        """
        Returns all runs without any filtering.

        Returns
        -------
        List[Run]
            A list of all runs.
        """
        return self.all_runs

    def filtered(self) -> List[Run]:
        """
        Returns runs that meet or exceed the specified sense value.

        Returns
        -------
        List[Run]
            A list of filtered runs.
        """
        return [run for run in self.all_runs if (run.end_index - run.start_index + 1) >= self.sense]

    def __repr__(self):
        """Provide a string representation of the RunList object."""
        return f"RunList(all_runs={self.all_runs}, sense={self.sense}, original_length={self.original_length})"


class RunLengthCodec:
    """
    A codec class for Run-Length Encoding (RLE) and decoding.

    Parameters
    ----------
    tol_merge : int, optional
        The tolerance value to merge close runs. Default is 20.
    remove_incomplete : bool, optional
        Whether to remove incomplete runs during sanitization. Default is False.
    merge_closed : bool, optional
        Whether to merge close runs during sanitization. Default is False.
    base_class : int, optional
        The base class value to be excluded from runs. Default is 0.

    Methods
    -------
    encode(x: List[int], sense: int) -> RunList:
        Encodes a list of integers into RLE runs.
    decode(encoded_runs: RunList) -> List[int]:
        Decodes RLE runs back into the original list of integers.
    sanitize(run_list: RunList) -> RunList:
        Cleans and merges runs based on the codec's parameters.
    __call__(data: Union[List[int], List[List[int]], RunList, List[RunList]], sense: Optional[int] = None) -> Union[RunList, List[RunList], List[int], List[List[int]]]:
        Encodes or decodes data based on the input type.
    """

    def __init__(
        self, tol_merge: int = 20, remove_incomplete: bool = False, merge_closed: bool = False, base_class: int = 0
    ):
        """
        Initialize the RunLengthCodec with specified parameters.

        Parameters
        ----------
        tol_merge : int, optional
            The tolerance value to merge close runs. Default is 20.
        remove_incomplete : bool, optional
            Whether to remove runs that start at index 0 or end at the last index. Default is False.
        merge_closed : bool, optional
            Whether to merge runs that are close to each other. Default is False.
        base_class : int, optional
            The base class value to exclude from runs. Runs with this value will not be encoded. Default is 0.
        """
        self.tol_merge = tol_merge
        self.remove_incomplete = remove_incomplete
        self.merge_closed = merge_closed
        self.base_class = base_class

    def encode(self, x: List[int], sense: int) -> RunList:
        """
        Encode a list of integers using Run-Length Encoding (RLE).

        This method converts a sequence of integers into a list of runs.
        Each run is represented as a `Run` object containing the start index,
        end index, and the value of the run. Runs with a length smaller than
        the specified `sense` are ignored in the `filtered()` method.

        Parameters
        ----------
        x : List[int]
            The input list of integers to be encoded.
        sense : int
            The minimum length of runs to be considered. Runs shorter than this
            value are excluded from the output when calling `filtered()`.

        Returns
        -------
        RunList
            A `RunList` object containing all runs, the sense value, and the original list length.
        """
        original_length = len(x)
        if not x:
            all_runs = []
        else:
            all_runs = []
            current_index = 0
            for value, group in groupby(x):
                group_list = list(group)
                run_length = len(group_list)
                start_index = current_index
                end_index = current_index + run_length - 1
                if value != self.base_class:
                    all_runs.append(Run(start_index=start_index, end_index=end_index, value=value))
                current_index += run_length
        run_list = RunList(all_runs=all_runs, sense=sense, original_length=original_length)
        sanitized_run_list = self.sanitize(run_list)
        return sanitized_run_list

    def decode(self, encoded_runs: RunList) -> List[int]:
        """
        Decode a list of RLE runs back to the original list of integers.

        This method reconstructs the original sequence of integers from a `RunList` object.
        Each `Run` specifies the start index, end index, and the value to be filled in that range.
        The length of the output list is determined by the `original_length` stored in `RunList`.

        Parameters
        ----------
        encoded_runs : RunList
            A `RunList` object containing runs to be decoded.

        Returns
        -------
        List[int]
            The decoded list of integers reconstructed from the runs.
        """
        if not encoded_runs.all_runs:
            return [self.base_class] * encoded_runs.original_length  # Return background if no runs

        decoded = [self.base_class] * encoded_runs.original_length  # Initialize with background label 0

        for run in encoded_runs.all_runs:
            on = max(run.start_index, 0)  # Ensure non-negative start index
            off = min(run.end_index, encoded_runs.original_length - 1)  # Clamp to maximum index
            cls = run.value

            # Assign the class label to the specified range
            for i in range(on, off + 1):
                decoded[i] = cls

        return decoded

    def sanitize(self, run_list: RunList) -> RunList:
        """
        Clean and merge runs based on the codec's parameters.

        This function processes the RunList by:
            1. Removing incomplete runs (if remove_incomplete is True).
            2. Merging close runs (if merge_closed is True), while respecting the base_class.

        Parameters
        ----------
        run_list : RunList
            The RunList object to be sanitized.

        Returns
        -------
        RunList
            The sanitized RunList object.
        """
        all_runs = run_list.all_runs.copy()

        # 1. Remove incomplete runs
        if self.remove_incomplete:
            # Incomplete runs are those that start at 0 or end at original_length -1
            sanitized_runs = [
                run for run in all_runs if run.start_index != 0 and run.end_index != run_list.original_length - 1
            ]
        else:
            sanitized_runs = all_runs

        # 2. Merge close runs
        if self.merge_closed and sanitized_runs:
            # Sort runs by start_index to ensure order
            sanitized_runs.sort(key=lambda run: run.start_index)

            merged_runs = []
            current_run = sanitized_runs[0]

            for next_run in sanitized_runs[1:]:
                gap = next_run.start_index - current_run.end_index - 1

                if gap <= self.tol_merge:
                    if current_run.value == next_run.value:
                        # Same value runs, merge them
                        current_run.end_index = next_run.end_index
                    else:
                        # Different value runs, decide which one to merge based on class
                        # Base class should not be involved in merging
                        if current_run.value == self.base_class or next_run.value == self.base_class:
                            # Do not merge with base class
                            merged_runs.append(current_run)
                            current_run = next_run
                        elif next_run.value < current_run.value:
                            # Merge current_run into next_run (retain next_run's value)
                            current_run.end_index = next_run.end_index
                            current_run.value = next_run.value
                        else:
                            # Merge next_run into current_run (retain current_run's value)
                            current_run.end_index = next_run.end_index
                else:
                    merged_runs.append(current_run)
                    current_run = next_run

            merged_runs.append(current_run)
            sanitized_runs = merged_runs

        return RunList(all_runs=sanitized_runs, sense=run_list.sense, original_length=run_list.original_length)

    def __call__(
        self, data: Union[List[int], List[List[int]], RunList, List[RunList]], sense: Optional[int] = None
    ) -> Union[RunList, List[RunList], List[int], List[List[int]]]:
        """
        Encode or decode data based on its type.

        If the input `data` is a list of integers, it encodes the list using RLE.
        If the input `data` is a list of lists of integers (batch), it encodes each list.
        If the input `data` is a `RunList`, it decodes it back to the original list of integers.
        If the input `data` is a list of `RunList` (batch), it decodes each `RunList`.

        Parameters
        ----------
        data : Union[List[int], List[List[int]], RunList, List[RunList]]
            The data to be encoded or decoded.
            - If `List[int]`, the data will be encoded.
            - If `List[List[int]]`, the data will be batch encoded.
            - If `RunList`, the data will be decoded.
            - If `List[RunList]`, the data will be batch decoded.
        sense : Optional[int], optional
            The minimum length of runs to be considered during encoding.
            Required if `data` is a `List[int]` or `List[List[int]]`.
            Ignored if `data` is a `RunList` or `List[RunList]`.
            Default is None.

        Returns
        -------
        Union[RunList, List[RunList], List[int], List[List[int]]]
            - Returns a `RunList` object if encoding a single list.
            - Returns a list of `RunList` objects if encoding a batch of lists.
            - Returns a list of integers if decoding a single `RunList`.
            - Returns a list of lists of integers if decoding a batch of `RunList`s.

        Raises
        ------
        ValueError
            If `sense` is not provided when encoding.
        TypeError
            If `data` is neither a list of integers, list of lists of integers, `RunList`, nor a list of `RunList`s.
        """
        if isinstance(data, list):
            if all(isinstance(item, list) for item in data):
                # Batch encoding
                if sense is None:
                    raise ValueError("Parameter 'sense' must be provided for encoding.")
                return [self.encode(sample, sense) for sample in data]
            elif all(isinstance(item, int) for item in data):
                # Single encoding
                if sense is None:
                    raise ValueError("Parameter 'sense' must be provided for encoding.")
                return self.encode(data, sense)
            else:
                raise TypeError("Input list must be a list of integers or a list of lists of integers.")
        elif isinstance(data, RunList):
            # Single decoding
            return self.decode(data)
        elif isinstance(data, list) and all(isinstance(item, RunList) for item in data):
            # Batch decoding
            return [self.decode(run_list) for run_list in data]
        else:
            raise TypeError(
                "Input data must be a list of integers, list of lists of integers, RunList, or list of RunList objects."
            )
