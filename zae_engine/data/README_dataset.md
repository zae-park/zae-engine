# Data Subpackage

The `data` subpackage of `zae-engine` provides essential utilities to efficiently handle data processing for deep learning projects using PyTorch. It contains extended implementations of PyTorch `Dataset` and collate functions to manage various data formats, especially when dealing with large-scale data.

## Overview

The `data` subpackage primarily consists of:

- **Dataset Extensions**: Custom datasets that facilitate working with non-standard data formats, such as Parquet, commonly used for large-scale data.
- **CollateBase for DataLoader**: A base class for preprocessing and batching data using a series of custom functions, which can be passed as the `collate_fn` to PyTorch's `DataLoader`.

## Features

### 1. ParquetDataset
The `ParquetDataset` class is a custom extension of PyTorch's `Dataset` to handle Parquet files, a common format for large-scale data processing. This dataset provides features such as:

- **Caching**: Supports loading multiple Parquet files into memory for efficient access, with customizable cache size.
- **Shuffling**: Provides shuffling options for indices to facilitate randomized data access.
- **Column Selection**: Allows selection of specific columns to be loaded from the Parquet files, making it possible to work with only the necessary data.

#### Example Usage
```python
from torch.utils.data import DataLoader
from zae_engine.data.dataset import ParquetDataset

# Example initialization of ParquetDataset
parquet_paths = ["data/file1.parquet", "data/file2.parquet"]
dataset = ParquetDataset(parquet_path=parquet_paths, fs=None, columns=("col1", "col2"))

# DataLoader with ParquetDataset
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

The `ParquetDataset` allows seamless integration with PyTorch's `DataLoader`, making it easy to handle large-scale Parquet data for deep learning tasks.

### 2. CollateBase
The `CollateBase` class is designed to handle complex preprocessing tasks while collating batches of data. It provides a modular way to apply multiple preprocessing functions to the data before batching.

- **Flexible Initialization**: Supports initialization with either an `OrderedDict` or a list of functions that will be applied in sequence to each batch.
- **Batch Processing**: Allows accumulation of batches by converting lists of dictionaries into a dictionary of lists, providing compatibility with PyTorch `DataLoader`'s `collate_fn`.
- **Input-Output Structure Checking**: Automatically checks if the registered preprocessing functions maintain the input-output structure of sample data, reducing the chance of data inconsistency.

#### Example Usage
```python
from collections import OrderedDict
from zae_engine.data.collate import CollateBase

def custom_fn1(batch):
    # Custom preprocessing function
    return batch

def custom_fn2(batch):
    # Another custom preprocessing function
    return batch

functions = OrderedDict([('fn1', custom_fn1), ('fn2', custom_fn2)])
collator = CollateBase(x_key=['x'], y_key=['y'], aux_key=['aux'], functions=functions)

# Using CollateBase with DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
```

With `CollateBase`, you can define a sequence of transformations that will be applied to each batch, making data preprocessing highly modular and customizable.

### 3. Preprocessing Functions for CollateBase
In addition to `CollateBase`, the `data/collate` module provides several preprocessing classes that can be used to extend the functionality of `CollateBase`. These classes include:

- **UnifiedChunker**: Splits input data into chunks based on its dimensionality, supporting both 1D and 2D tensors. Useful for sequence data that needs to be divided into smaller segments.
- **Chunk**: Reshapes tensors within a batch for specific chunk sizes.
- **HotEncoder**: Converts labels into one-hot encoded format for easier classification tasks.
- **SignalFilter**: Applies signal filtering (e.g., bandpass, lowpass) to the data, which is useful for preprocessing time-series or sensor data.
- **Spliter**: Splits signals within a batch with optional overlapping segments, helpful for creating overlapping sequences.
- **SignalScaler**: Scales signal data using MinMaxScaler to normalize values, ensuring consistent data ranges.

#### Example of Using Preprocessing Functions
```python
from zae_engine.data.collate import CollateBase, HotEncoder, SignalFilter

# Define preprocessing functions
encoder = HotEncoder(n_cls=10)
filter_fn = SignalFilter(fs=100.0, method='bandpass', lowcut=0.5, highcut=30.0)

collator = CollateBase(functions=[encoder, filter_fn])

# DataLoader with custom collator
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
```
These preprocessing functions allow users to flexibly handle complex transformations on their input data, making `CollateBase` a powerful tool for customizing the batching process.

## Summary
The `data` subpackage of `zae-engine` is designed to extend the capabilities of PyTorch's data processing tools by providing:
- Custom datasets like `ParquetDataset` for handling large-scale data formats.
- A flexible `CollateBase` class for modular preprocessing of data batches.
- Several built-in preprocessing functions to use with `CollateBase` for tasks like chunking, encoding, and filtering.

These tools help streamline the data preparation process, particularly for large-scale or specialized data formats, and facilitate modular preprocessing workflows that integrate seamlessly with PyTorch's `DataLoader`.

