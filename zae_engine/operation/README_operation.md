# Operation Subpackage

The `operation` subpackage provides a collection of utilities and modules for complex algorithm-based operations often needed for data preprocessing and feature engineering. These operations include morphological transformations, run-length encoding, and other useful operations for working with 1D data. This subpackage allows for convenient implementation of advanced data manipulation techniques, ensuring code clarity and consistency.

## Features

### 1. Morphological Operations

The subpackage provides a `MorphologicalLayer` class for performing morphological operations on 1D data. These operations can be used for signal processing or feature extraction, applying transformations like opening and closing.

- **Class**: `MorphologicalLayer`
- **Description**: Applies morphological operations (e.g., dilation and erosion) on 1D tensors.
- **Operations Supported**:
  - **Closing** (`c`): Dilation followed by erosion.
  - **Opening** (`o`): Erosion followed by dilation.
- **Parameters**:
  - `ops` (str): A string where each character represents an operation ('c' for closing, 'o' for opening).
  - `window_size` (List[int]): A list of window sizes for each operation.

Example usage:
```python
from zae_engine.operation import MorphologicalLayer
layer = MorphologicalLayer(ops='co', window_size=[3, 5])
output = layer(input_tensor)
```

### 2. Run-Length Encoding and Decoding

Run-length encoding is a powerful technique for compressing sequences of values. The `RunLengthCodec` class offers methods to encode and decode lists of values while providing efficient handling for runs of repeated values.

- **Class**: `RunLengthCodec`
- **Description**: Encodes and decodes sequences using run-length encoding (RLE).
- **Attributes**:
  - `tol_merge` (int): Tolerance value for merging close runs.
  - `remove_incomplete` (bool): Whether to remove incomplete runs.
  - `merge_closed` (bool): Whether to merge close runs.
  - `base_class` (int): Base class value to exclude from runs.
- **Methods**:
  - `encode(x, sense)`: Encodes a list of integers into RLE runs.
  - `decode(encoded_runs)`: Decodes RLE runs back into the original list.
  - `sanitize(run_list)`: Cleans and merges runs based on parameters.

Example usage:
```python
from zae_engine.operation import RunLengthCodec
codec = RunLengthCodec(tol_merge=10, remove_incomplete=True)
encoded = codec.encode([1, 1, 2, 2, 0, 0, 3, 3], sense=2)
decoded = codec.decode(encoded)
```

### 3. Nearest Value Search

The `arg_nearest` function finds the index of the nearest value in a given sorted array, which is useful for signal alignment or time-series processing.

- **Function**: `arg_nearest`
- **Description**: Finds the index of the nearest value in a sorted array to a given reference value.
- **Parameters**:
  - `arr` (Union[np.ndarray, torch.Tensor]): The input sorted array.
  - `value` (int): The reference value to find the nearest element for.
  - `return_value` (bool): Whether to return the nearest value along with its index.

Example usage:
```python
import numpy as np
from zae_engine.operation import arg_nearest
arr = np.array([1, 3, 5, 7, 9])
index, nearest_value = arg_nearest(arr, 6)
```

### 4. Run Representation

The subpackage also includes the `Run` and `RunList` classes for representing the results of run-length encoding, making it easy to work with segments of data that have similar properties.

- **Class**: `Run`
  - **Attributes**: `start_index`, `end_index`, `value`.
- **Class**: `RunList`
  - **Attributes**: `all_runs`, `sense`, `original_length`.
  - **Methods**: `raw()`, `filtered()` for accessing all or filtered runs.

Example usage:
```python
from zae_engine.operation import Run, RunList
run = Run(start_index=0, end_index=5, value=1)
run_list = RunList(all_runs=[run], sense=2, original_length=10)
filtered_runs = run_list.filtered()
```

## Summary

The `operation` subpackage includes powerful tools for data preprocessing and feature engineering. These tools are useful for tasks like run-length encoding, finding nearest values, and applying morphological transformations to 1D signals. Whether you need to work with compressed data representations or complex morphological operations, the `operation` subpackage provides efficient, reusable solutions to speed up your data manipulation workflows.