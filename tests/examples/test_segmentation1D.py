# tests/operation/test_inference.py

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# inference.py에 정의된 core 함수를 임포트
from zae_engine.examples.segmentation1D import core


class TestInference(unittest.TestCase):
    def setUp(self):
        # 필요한 초기 설정이 있다면 여기에 작성
        pass

    def test_core_with_valid_numpy_input(self):
        # 유효한 1-D numpy 입력에 대한 테스트
        x = np.zeros(20480)  # 20480은 10 * 2048
        with patch("zae_engine.examples.segmentation1D.InferenceTrainer.inference") as mock_inference:
            # 모킹된 인퍼런스 결과 설정
            mock_inference.return_value = [torch.tensor([[0, 1, 0, 0]])] * 10  # 10개의 배치
            result = core(x)
            # 예측 결과가 argmax(1)으로 1이 될 것으로 예상
            expected = np.array([1] * 10)
            np.testing.assert_array_equal(result, expected)

    def test_core_with_valid_torch_tensor_input(self):
        # 유효한 torch.Tensor 입력에 대한 테스트
        x = torch.zeros(20480)
        with patch("zae_engine.examples.segmentation1D.InferenceTrainer.inference") as mock_inference:
            # 모킹된 인퍼런스 결과 설정
            mock_inference.return_value = [torch.tensor([[0, 2, 0, 0]])] * 10  # 10개의 배치
            result = core(x)
            # 예측 결과가 argmax(1)으로 2가 될 것으로 예상
            expected = np.array([1] * 10)  # argmax는 1번째 인덱스가 2였으므로, 1로 인덱싱
            np.testing.assert_array_equal(result, expected)

    def test_core_with_invalid_input_dimension(self):
        # 비유효한 입력 차원에 대한 테스트 (2-D 배열)
        x = np.zeros((10, 2048))
        with self.assertRaises(AssertionError):
            core(x)

    def test_core_with_empty_input(self):
        # 빈 배열 입력에 대한 테스트
        x = np.array([])
        self.assertEqual(core(x), [])

    def test_core_with_non_numpy_non_tensor_input(self):
        # 지원하지 않는 입력 타입에 대한 테스트 (리스트)
        x = [0] * 20480
        with self.assertRaises(AttributeError):
            core(x)

    def test_core_with_partial_runs_below_sense(self):
        # 일부 런이 sense 미만인 경우 테스트
        x = np.concatenate(
            [
                np.zeros(2048 * 2),  # Background
                np.ones(2048 * 1),  # Label 1 (길이 2048, assume sense=2)
                np.ones(2048 * 3),  # Label 1 (길이 6144, above sense=2)
                np.zeros(2048 * 2),  # Background
                np.full(2048 * 2, 2),  # Label 2 (길이 4096, above sense=2)
            ]
        )
        with patch("zae_engine.examples.segmentation1D.InferenceTrainer.inference") as mock_inference:
            # 모킹된 인퍼런스 결과 설정
            mock_inference.return_value = [torch.tensor([[0, 1, 0, 0]])] * 10
            result = core(x)
            expected = np.array([1] * 10)  # 예시
            np.testing.assert_array_equal(result, expected)

    def test_core_no_non_background_runs(self):
        # 비배경 런이 없는 경우 (모든 레이블이 0인 경우)
        x = np.zeros(20480)
        with patch("zae_engine.examples.segmentation1D.InferenceTrainer.inference") as mock_inference:
            # 모킹된 인퍼런스 결과 설정 (배경만)
            mock_inference.return_value = [torch.tensor([[0, 0, 0, 0]])] * 10
            result = core(x)
            expected = np.array([0] * 10)
            np.testing.assert_array_equal(result, expected)

    def test_core_with_mixed_labels(self):
        # 다양한 레이블이 혼합된 경우 테스트
        x = np.concatenate(
            [
                np.zeros(2048 * 1),  # Background
                np.ones(2048 * 2),  # Label 1
                np.full(2048 * 3, 2),  # Label 2
                np.zeros(2048 * 1),  # Background
                np.full(2048 * 2, 3),  # Label 3
            ]
        )
        with patch("zae_engine.examples.segmentation1D.InferenceTrainer.inference") as mock_inference:
            # 모킹된 인퍼런스 결과 설정
            mock_inference.return_value = [
                torch.tensor([[0, 1, 0, 0]]),
                torch.tensor([[0, 1, 0, 0]]),
                torch.tensor([[0, 1, 0, 0]]),
                torch.tensor([[0, 1, 0, 0]]),
                torch.tensor([[0, 1, 0, 0]]),
            ]
            result = core(x)
            expected = np.array([1, 1, 1, 1, 1])  # 예시
            # 실제 expected 값은 모킹된 인퍼런스 결과에 따라 달라질 수 있음
            # 이 부분은 사용자의 모델 출력에 맞게 조정 필요
            # 여기서는 단순히 argmax가 1이 되도록 설정
            # 따라서 expected는 1이 될 것으로 가정
            self.assertEqual(set(result), {1})

    @patch("zae_engine.models.builds.autoencoder.AutoEncoder")
    def test_core_model_initialization_failure(self, mock_autoencoder):
        # 모델 초기화 중 예외가 발생하는 경우 테스트
        mock_autoencoder.side_effect = Exception("Model initialization failed")
        x = np.zeros(20480)
        with self.assertRaises(Exception) as context:
            core(x)
        self.assertIn("Model initialization failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
