import time
from collections import defaultdict, namedtuple
from typing import Type, Union

import numpy as np
import pandas as pd
import fastparquet
from torch.utils import data


# class ParquetDataset(data.Dataset):
#     def __init__(
#         self,
#         parquet_path: Union[list | tuple],
#         fs,
#         raw_cols: tuple[str] = (),
#         use_cols: tuple[str] = (),
#         num_cached_parquet=5,
#         shuffle: bool = False,
#         batched_output: bool = False,
#     ):
#
#         self.fs = fs
#         self.shuffle = shuffle
#         self.batched_output = batched_output
#         self.row_canvas = namedtuple("row", field_names=use_cols)
#         self.raw_cols = raw_cols  # parquet file에서 사용할 column
#         self.use_cols = use_cols  # 출력 sample에서 사용할 column
#
#         # self.parquet_list = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
#         self.parquet_list = parquet_path
#         self.num_cached_parquet = num_cached_parquet  # 캐시할 파일 개수
#
#         self.steps_cache = int(np.ceil(len(self.parquet_list) / self.num_cached_parquet))  # cache step
#         self.current_parquet_idx = 0
#         self.current_pd_parquets = None  # cached parquets
#         self.current_indices_in_cache = []  # data index in cached parquet
#         self.steps_per_epoch = 0
#         self.total_len = self.get_total_length()
#
#         self._cache_setting()
#
#     def _cache_setting(self):
#         cur_pd, cur_indices = self._cache_parquet(self.current_parquet_idx)
#         self.current_pd_parquets = cur_pd
#         self.current_indices_in_cache = cur_indices
#
#     def get_total_length(self):
#         fdf = fastparquet.ParquetFile(self.parquet_list, open_with=self.fs.open)
#         total_len = 0
#         for df in fdf.iter_row_groups(columns=["unique_id"]):
#             total_len += len(df)
#         return total_len
#
#     def _cache_parquet(self, idx):
#         next_idx = (idx + 1) * self.num_cached_parquet
#         next_idx = None if next_idx > len(self.parquet_list) else next_idx
#
#         list_part_parquet = self.parquet_list[idx * self.num_cached_parquet : next_idx]
#
#         fparquet = fastparquet.ParquetFile(list_part_parquet, open_with=self.fs.open)
#
#         list_df = []
#         for df, fpar in zip(fparquet.iter_row_groups(columns=self.raw_cols), fparquet):
#             list_df.append(df)
#
#         df_data = pd.concat(list_df)
#         now = time.time()
#         seed = int((now - int(now)) * 1e5)
#         rng = np.random.RandomState(seed=seed)
#         np_indices = rng.permutation(len(df_data)) if self.shuffle else np.arange(len(df_data))
#         list_indices = np_indices.tolist()
#
#         return df_data, list_indices
#
#     def _transform_raw_to_array(self, pd_raw_data):
#         # pd_raw_data['request_list'] = torch.zeros((16, 16), dtype=torch.float32)
#         # if pd_raw_data['request_list'] is not None:
#         #     pass
#         # else:
#         #     pd_raw_data['request_list'] = []
#         sample = self.row_canvas(**pd_raw_data)
#         return sample
#
#     def __len__(self):
#         return self.total_len
#
#     def __getitem__(self, idx):
#         # https://d2.naver.com/helloworld/3773258
#         # idx는 사용하지 않고 parquet data queue에서 pop
#
#         refresh_idx = 1
#         # 현재 캐시 파일 내 데이터를 모두 탐색한 경우
#         if len(self.current_indices_in_cache) < refresh_idx:
#             # 다음 parquet 파일들을 로딩
#             self.current_parquet_idx += 1
#
#             # 모든 parquet 파일 로딩한 경우 첫 번째 파일로
#             if self.current_parquet_idx >= self.steps_cache:
#                 self.current_parquet_idx = 0
#
#             # parquet cache loading
#             self._cache_setting()
#
#         pd_idx = self.current_indices_in_cache.pop()
#         pd_raw = self.current_pd_parquets.iloc[pd_idx]
#
#         sample = self._transform_raw_to_array(pd_raw)
#
#         return sample
