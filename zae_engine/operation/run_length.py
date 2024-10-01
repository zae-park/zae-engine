from dataclasses import dataclass
from typing import List, Optional
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
    """

    def __init__(self, all_runs: List[Run], sense: int):
        """
        Initializes the RunList object.

        Parameters
        ----------
        all_runs : List[Run]
            A list of all runs obtained from encoding.
        sense : int
            The minimum length of runs to be considered in filtering.
        """
        self.all_runs = all_runs
        self.sense = sense

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


class RunLengthCodec:
    """
    A codec class for Run-Length Encoding (RLE) and decoding.

    Methods
    -------
    encode(x: List[int], sense: int) -> RunList:
        Encodes a list of integers into RLE runs.
    decode(encoded_runs: RunList, length: Optional[int] = 2500) -> List[int]:
        Decodes RLE runs back into the original list of integers.
    """

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
            A `RunList` object containing all runs and the sense value.
        """
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
                all_runs.append(Run(start_index=start_index, end_index=end_index, value=value))
                current_index += run_length
        return RunList(all_runs=all_runs, sense=sense)

    def decode(self, encoded_runs: RunList, length: Optional[int] = 2500) -> List[int]:
        """
        Decode a list of RLE runs back to the original list of integers.

        This method reconstructs the original sequence of integers from a `RunList` object.
        Each `Run` specifies the start index, end index, and the value to be filled in that range.
        The `length` parameter defines the total length of the output list.
        If a run's end index exceeds the specified length, it is clamped to the maximum index.

        Parameters
        ----------
        encoded_runs : RunList
            A `RunList` object containing runs to be decoded.
        length : int, optional
            The length of the output list. This should be greater than or equal
            to the maximum end index in the runs. Default is 2500.

        Returns
        -------
        List[int]
            The decoded list of integers reconstructed from the runs.
        """
        if not encoded_runs.all_runs:
            return [0] * length  # Return background if no runs

        max_index = max(run.end_index for run in encoded_runs.all_runs)
        required_length = max(max_index + 1, length)
        decoded = [0] * required_length  # Initialize with background label 0

        for run in encoded_runs.all_runs:
            on = run.start_index if run.start_index >= 0 else 0  # Ensure non-negative start index
            off = run.end_index
            cls = run.value

            if off >= required_length:
                off = required_length - 1  # Clamp to maximum index

            # Assign the class label to the specified range
            for i in range(on, off + 1):
                decoded[i] = cls

        # Truncate to specified length if necessary
        decoded = decoded[:length]

        return decoded
