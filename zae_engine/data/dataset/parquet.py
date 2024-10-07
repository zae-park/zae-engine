import time
from typing import List, Tuple, Union
from bisect import bisect_right

import numpy as np
import pandas as pd
import fastparquet
from torch.utils import data


class ParquetDataset(data.Dataset):
    """
    Custom PyTorch Dataset for loading and accessing data from multiple Parquet files efficiently.

    This dataset handles multiple Parquet files by caching them and provides indexing to access individual samples.
    It supports shuffling of data and selecting specific columns for use.

    Args:
        parquet_paths (List[str] | Tuple[str, ...]):
            List or tuple of paths to Parquet files.
        fs:
            Filesystem object (e.g., fsspec filesystem) to handle file operations.
        raw_cols (Tuple[str, ...], optional):
            Columns to read from the Parquet files. Defaults to ().
        use_cols (Tuple[str, ...], optional):
            Columns to include in the output samples. Defaults to ().
        shuffle (bool, optional):
            Whether to shuffle the dataset indices. Defaults to False.
    """

    def __init__(
        self,
        parquet_paths: Union[List[str], Tuple[str, ...]],
        fs,
        raw_cols: Tuple[str, ...] = (),
        use_cols: Tuple[str, ...] = (),
        shuffle: bool = False,
    ):
        self.fs = fs
        self.shuffle = shuffle
        self.raw_cols = raw_cols
        self.use_cols = list(use_cols)

        self.parquet_list = parquet_paths

        # compute cumulated rows
        self.cumulative_sizes = []
        total = 0
        self.parquet_sizes = []
        for parquet_file in self.parquet_list:
            fdf = fastparquet.ParquetFile(parquet_file, open_with=self.fs.open)
            num_rows = fdf.info["rows"]
            total += num_rows
            self.cumulative_sizes.append(total)
            self.parquet_sizes.append(num_rows)
        self.total_len = total

        # create index & shuffle
        self.indices = np.arange(self.total_len)
        if self.shuffle:
            np.random.shuffle(self.indices)

        # initial cache
        self.current_parquet_idx = None
        self.current_pd_parquets = None

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.total_len

    def __getitem__(self, idx):
        """
        Retrieves the sample corresponding to the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            dict:
                A dictionary containing the requested sample's data.
        """
        actual_idx = self.indices[idx]
        parquet_idx = bisect_right(self.cumulative_sizes, actual_idx)
        if parquet_idx == 0:
            row_idx = actual_idx
        else:
            row_idx = actual_idx - self.cumulative_sizes[parquet_idx - 1]

        # update cache if current parquet is different with cached.
        if parquet_idx != self.current_parquet_idx:
            self.current_parquet_idx = parquet_idx
            self._cache_setting(parquet_idx)

        # get row
        pd_raw = self.current_pd_parquets.iloc[row_idx]
        sample = pd_raw[self.use_cols].to_dict()
        return sample

    def _cache_setting(self, parquet_idx):
        """
        Loads the specified Parquet file into cache.

        Args:
            parquet_idx (int):
                Index of the Parquet file to load.
        """
        parquet_file = self.parquet_list[parquet_idx]
        fparquet = fastparquet.ParquetFile(parquet_file, open_with=self.fs.open)
        list_df = [df for df in fparquet.iter_row_groups(columns=self.raw_cols)]
        self.current_pd_parquets = pd.concat(list_df, ignore_index=True)
