# parquet_dataset.py

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
        columns (List[str], optional):
            Columns to read from the Parquet files. Defaults to None, which reads all columns.
        shuffle (bool, optional):
            Whether to shuffle the dataset indices. Defaults to False.
    """

    def __init__(
        self, parquet_paths: Union[List[str], Tuple[str, ...]], fs, columns: List[str] = None, shuffle: bool = False
    ):
        self.fs = fs
        self.shuffle = shuffle
        self.columns = columns  # 읽어올 컬럼 리스트

        self.parquet_list = parquet_paths

        # 누적 행 수 계산
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

        # 인덱스 생성 및 셔플
        self.indices = np.arange(self.total_len)
        if self.shuffle:
            np.random.shuffle(self.indices)

        # 캐시 초기화
        self.current_parquet_idx = None
        self.current_pd_parquets = None

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.total_len

    def __getitem__(self, idx: int) -> dict:
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

        # 현재 캐시된 Parquet 파일과 다르면 캐시를 갱신
        if parquet_idx != self.current_parquet_idx:
            self.current_parquet_idx = parquet_idx
            self._cache_setting(parquet_idx)

        # 해당 행 가져오기
        pd_raw = self.current_pd_parquets.iloc[row_idx]
        sample = pd_raw.to_dict()  # 모든 컬럼을 딕셔너리로 반환
        return sample

    def _cache_setting(self, parquet_idx: int):
        """
        Loads the specified Parquet file into cache.

        Args:
            parquet_idx (int):
                Index of the Parquet file to load.
        """
        parquet_file = self.parquet_list[parquet_idx]
        fparquet = fastparquet.ParquetFile(parquet_file, open_with=self.fs.open)
        list_df = (
            [df for df in fparquet.iter_row_groups(columns=self.columns)]
            if self.columns
            else [df for df in fparquet.iter_row_groups()]
        )
        self.current_pd_parquets = pd.concat(list_df, ignore_index=True)
