# Utils Sub-package Overview

The **utils** sub-package in the **zae-engine** library provides essential utility functions and decorators for deep learning workflows. These utilities simplify common tasks like loading external data, performing conversions between data types, ensuring tensor shape consistency, and measuring execution times. The utils sub-package is a critical component to enhance usability and streamline the implementation of the **zae-engine**.

## Utility Functions

### 1. Data I/O Utilities

- **image_from_url(url: str, save_dst: str = None) -> Union[None, Image.Image]**
  - Downloads an image from a given URL, optionally saving it to a specified location or returning it as a PIL Image object.

- **example_ecg(beat_idx: int = None) -> Tuple[np.ndarray, ...]**
  - Loads a 10-second ECG recording from the LUDB dataset. Users can specify an index to extract a specific beat or retrieve the entire sequence.

- **example_mri() -> nibabel.arrayproxy.ArrayProxy**
  - Loads a 4D MRI scan using the NIfTI file format. This function returns the MRI data as an `ArrayProxy` object for efficient memory usage.

## Decorators

The sub-package includes several decorators to aid in handling data conversions, validating tensor shapes, and measuring execution times. These decorators make the code cleaner, easier to maintain, and reusable.

### 1. Conversion Decorators

- **np2torch(dtype: torch.dtype, *keys: str, n: Optional[int] = None) -> Callable**
  - Converts specified NumPy arrays to PyTorch tensors with the given data type. It can handle entire arguments or specific dictionary keys.

- **torch2np(dtype: Optional[np.dtype] = None, *keys: str, n: Optional[int] = None) -> Callable**
  - Converts specified PyTorch tensors to NumPy arrays. This decorator can be applied to either function arguments or values in a dictionary.

### 2. Validation Decorator

- **shape_check(*keys: Union[int, str]) -> Callable**
  - Ensures that the shapes of specified arguments are consistent. This helps prevent errors due to shape mismatches during computations.

### 3. Timing Decorator

- **tictoc(func: Callable) -> Callable**
  - Measures and prints the elapsed time for a function or method. This is particularly useful for identifying performance bottlenecks.

## Usage Example

Below is an example showcasing how to use the utilities and decorators provided by the **utils** sub-package.

```python
from zae_engine.utils import image_from_url, example_ecg, example_mri
from zae_engine.utils.decorators import np2torch, tictoc
import torch
import numpy as np

# Example: Using the image_from_url utility
downloaded_image = image_from_url('https://example.com/sample_image.png')

# Example: Using np2torch decorator
@np2torch(torch.float32, 'input')
def model_inference(input: np.ndarray):
    # The 'input' will be automatically converted to a torch.Tensor
    return input

# Example: Using tictoc to measure function execution time
@tictoc
def compute_heavy_task():
    time.sleep(2)  # Simulating a time-consuming task

compute_heavy_task()  # Prints elapsed time
```

## Summary

The **utils** sub-package of **zae-engine** provides versatile utilities and decorators to assist in data handling, type conversion, and function timing. It plays an essential role in simplifying the development and training of machine learning models by providing easy-to-use tools and efficient helpers for common tasks.