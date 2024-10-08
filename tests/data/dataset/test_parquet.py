import unittest
import os
import tempfile
from typing import List

import pandas as pd
import fastparquet
import fsspec
from torch.utils.data import DataLoader

from zae_engine.data.dataset import ParquetDataset


class TestParquetDataset(unittest.TestCase):
    def setUp(self):
        # 임시 디렉토리 생성
        self.test_dir = tempfile.TemporaryDirectory()
        self.parquet_paths = self._create_parquet_files()

        # 파일 시스템 초기화
        self.fs = fsspec.filesystem("file")

        # 사용할 컬럼 정의
        self.columns = ["id", "feature1", "feature2"]  # 리스트 형태로 변경

    def tearDown(self):
        # 임시 디렉토리 정리
        self.test_dir.cleanup()

    def _create_parquet_files(self) -> List[str]:
        """
        테스트용 Parquet 파일 생성.

        Returns:
            List[str]: 생성된 Parquet 파일 경로 목록.
        """
        parquet_files = []
        for i in range(3):  # 3개의 Parquet 파일
            data = {
                "id": range(i * 100, (i + 1) * 100),
                "feature1": [f"feat1_{j}" for j in range(i * 100, (i + 1) * 100)],
                "feature2": [f"feat2_{j}" for j in range(i * 100, (i + 1) * 100)],
                "unused_col": [f"unused_{j}" for j in range(i * 100, (i + 1) * 100)],
            }
            df = pd.DataFrame(data)
            parquet_path = os.path.join(self.test_dir.name, f"test_{i}.parquet")
            fastparquet.write(parquet_path, df, compression="SNAPPY")
            parquet_files.append(parquet_path)
        return parquet_files

    def test_length(self):
        """
        데이터셋의 전체 길이가 올바른지 테스트.
        """
        dataset = ParquetDataset(parquet_paths=self.parquet_paths, fs=self.fs, columns=self.columns, shuffle=False)
        expected_length = 300  # 3 파일 * 100 행
        self.assertEqual(len(dataset), expected_length, "Dataset length should be 300")

    def test_get_item(self):
        """
        특정 인덱스의 샘플이 올바르게 반환되는지 테스트.
        """
        dataset = ParquetDataset(parquet_paths=self.parquet_paths, fs=self.fs, columns=self.columns, shuffle=False)
        # 첫 번째 샘플 테스트
        sample = dataset[0]
        expected = {"id": 0, "feature1": "feat1_0", "feature2": "feat2_0"}
        self.assertEqual(sample, expected, "First sample does not match expected values")

        # 마지막 샘플 테스트
        sample = dataset[299]
        expected = {"id": 299, "feature1": "feat1_299", "feature2": "feat2_299"}
        self.assertEqual(sample, expected, "Last sample does not match expected values")

    def test_shuffle(self):
        """
        데이터셋이 셔플될 때 순서가 변경되는지 테스트.
        """
        dataset1 = ParquetDataset(parquet_paths=self.parquet_paths, fs=self.fs, columns=self.columns, shuffle=True)
        dataset2 = ParquetDataset(parquet_paths=self.parquet_paths, fs=self.fs, columns=self.columns, shuffle=True)
        # 두 데이터셋의 첫 번째 샘플이 다를 확률이 높습니다.
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        # 동일할 경우, 다른 샘플을 비교
        if sample1 == sample2:
            sample1 = dataset1[1]
            sample2 = dataset2[1]
        self.assertNotEqual(sample1, sample2, "Shuffled datasets should have different sample orders")

    def test_dataloader_integration(self):
        """
        DataLoader와의 통합 테스트.
        """
        dataset = ParquetDataset(parquet_paths=self.parquet_paths, fs=self.fs, columns=self.columns, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
        batch_count = 0
        for batch in dataloader:
            # 각 키에 대한 샘플 수 확인
            for key in self.columns:
                self.assertIn(key, batch, f"Batch should contain key '{key}'")
                self.assertEqual(
                    len(batch[key]), 50, f"Each batch should have 50 samples for key '{key}', got {len(batch[key])}"
                )
            batch_count += 1
        self.assertEqual(batch_count, 6, "There should be 6 batches (300 / 50)")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
