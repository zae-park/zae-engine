import unittest

import torch
import numpy as np

from zae_engine.metrics.confusion import confusion_matrix, print_confusion_matrix
from zae_engine.operation import MorphologicalLayer, label_to_onoff, onoff_to_label, Run, run_length_encoding


class TestMorphology(unittest.TestCase):
    def setUp(self) -> None:
        self.zit_tensor = torch.tensor(np.array([1] * 80 + [0, 1] * 20 + [0] * 80), dtype=torch.float32).reshape(
            1, 1, -1
        )
        self.open_close = MorphologicalLayer("CO", [9, 9])
        self.close_open = MorphologicalLayer("OC", [9, 9])
        super().setUp()

    def test_morph(self):
        res1 = self.open_close(self.zit_tensor)
        res2 = self.close_open(self.zit_tensor)

        self.assertEqual(res1.shape, res2.shape)
        self.assertGreaterEqual(res1.sum(), res2.sum())


class TestDrawConfusion(unittest.TestCase):

    def setUp(self) -> None:
        self.num_classes = 30
        self.yhat_ex = np.random.randint(0, self.num_classes, 2000)
        self.y_ex = np.random.randint(0, self.num_classes, 2000)
        self.no_elements = np.array([])
        return super().setUp()

    def check_shape_and_classes(self):
        cm = confusion_matrix(self.y_ex, self.yhat_ex, num_classes=self.num_classes)
        classes, _ = cm.shape

        self.assertEqual(classes, self.num_classes)
        self.assertEqual((self.num_classes, self.num_classes), cm.shape)

    def test_zero_num_classes(self):
        with self.assertRaises(Exception):
            confusion_matrix(self.y_ex, self.yhat_ex, num_classes=0)


class TestPrintConfusion(unittest.TestCase):

    def check_input_as_ndarray(self):
        with self.assertRaises(Exception):
            print_confusion_matrix(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))


class TestSegmentCodec(unittest.TestCase):
    def test_label_to_onoff_single(self):
        # Test single data sample
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
        expected_onoff = [[2, 4, 1], [6, 7, 2]]  # Label 1 from index 2 to 4  # Label 2 from index 6 to 7
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_multiple(self):
        # Test multiple data samples
        labels = np.array([[0, 0, 1, 1, 1, 0, 2, 2], [0, 1, 1, 2, 2, 1, 0, 0]])
        expected_onoff = [[[2, 4, 1], [6, 7, 2]], [[1, 2, 1], [3, 4, 2]]]
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_sensitivity(self):
        # Test sensitivity parameter
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0])
        # With sense=3, runs shorter than 3 should be ignored
        expected_onoff = [[5, 7, 1]]
        result = label_to_onoff(labels, sense=3, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_middle_only(self):
        # Test middle_only parameter
        labels = np.array([1, 1, 0, 1, 1])
        # middle_only=True should ignore the first and last runs
        expected_onoff = []
        result = label_to_onoff(labels, sense=2, middle_only=True, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_onoff_to_label_single(self):
        # Test single data sample
        onoff = np.array([[2, 4, 1], [6, 7, 2]])
        expected_labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    def test_onoff_to_label_multiple(self):
        # Test multiple runs
        onoff = np.array([[1, 2, 1], [3, 4, 2]])
        expected_labels = np.array([0, 1, 1, 2, 2, 0, 0, 0])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    def test_onoff_to_label_nan(self):
        # Test with np.nan for boundaries
        onoff = np.array(
            [
                [np.nan, 3, 1],  # Start is undefined, set from 0 to 3
                [4, np.nan, 2],  # End is undefined, set from 4 to length-1
            ]
        )
        expected_labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    def test_onoff_to_label_invalid_shape(self):
        # Test invalid onoff shape
        onoff = np.array([0, 4, 1])  # Should be 2D
        with self.assertRaises(ValueError):
            onoff_to_label(onoff, length=8)

    def test_label_to_onoff_invalid_shape(self):
        # Test invalid labels shape
        labels = np.array([])  # Empty array
        expected_onoff = []
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_tensor_input(self):
        # Test torch.Tensor input
        labels = torch.tensor([0, 0, 1, 1, 1, 0, 2, 2])
        expected_onoff = [[2, 4, 1], [6, 7, 2]]
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_onoff_to_label_tensor_input(self):
        # Test torch.Tensor input
        onoff = torch.tensor([[2, 4, 1], [6, 7, 2]])
        expected_labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    # 추가 테스트 케이스
    def test_label_to_onoff_no_runs(self):
        # Test labels with no non-background runs
        labels = np.array([0, 0, 0, 0])
        expected_onoff = []
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_partial_runs(self):
        # Test labels where some runs are below sensitivity
        labels = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 3, 0])
        expected_onoff = [[1, 2, 1], [4, 5, 2], [7, 9, 3]]
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=np.nan)
        self.assertEqual(result, expected_onoff)


class TestRunLengthEncoding(unittest.TestCase):
    def test_empty_list(self):
        """입력 리스트가 비어있는 경우"""
        x = []
        sense = 2
        expected = []
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_all_runs_above_sense(self):
        """모든 run이 sense보다 큰 경우"""
        x = [1, 1, 2, 2, 2, 3, 3, 3]
        sense = 2
        expected = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=4, value=2),
            Run(start_index=5, end_index=7, value=3),
        ]
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_runs_below_sense_no_merge(self):
        """sense보다 작은 run이 있지만, 양 옆 run이 다른 값인 경우 병합되지 않음"""
        x = [1, 1, 2, 3, 1, 1]
        sense = 2
        expected = [Run(start_index=0, end_index=1, value=1), Run(start_index=4, end_index=5, value=1)]
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_runs_below_sense_with_merge(self):
        """sense보다 작은 run이 있고, 양 옆 run이 같은 값인 경우 병합"""
        # 병합 로직이 제거되었으므로, 이 테스트는 더 이상 유효하지 않습니다.
        # 기대 결과는 양 옆 run을 병합하지 않고, 중간 run을 제외한 run들만 포함합니다.
        x = [1, 1, 2, 1, 1]
        sense = 2
        expected = [Run(start_index=0, end_index=1, value=1), Run(start_index=3, end_index=4, value=1)]
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_multiple_merges(self):
        """여러 개의 작은 run이 있고, 이를 통해 여러 번 병합이 일어나는 경우"""
        # 병합 로직이 제거되었으므로, 모든 run은 독립적으로 처리됩니다.
        x = [1, 1, 2, 1, 1, 2, 1, 1]
        sense = 2
        expected = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=3, end_index=4, value=1),
            Run(start_index=6, end_index=7, value=1),
        ]
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_ignored_run_at_start(self):
        """무시된 run이 리스트의 시작 부분에 있는 경우"""
        x = [2, 2, 1, 1, 3, 3]
        sense = 3
        expected = []
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_ignored_run_at_end(self):
        """무시된 run이 리스트의 끝 부분에 있는 경우"""
        x = [1, 1, 3, 3]
        sense = 3
        expected = []
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_all_runs_below_sense(self):
        """모든 run이 sense보다 작은 경우"""
        x = [1, 1, 2, 2, 3, 3]
        sense = 3
        expected = []
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_mixed_runs(self):
        """sense보다 작은 run과 큰 run이 혼합된 경우"""
        x = [1, 1, 2, 2, 2, 3, 1, 1, 4]
        sense = 2
        expected = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=4, value=2),
            Run(start_index=6, end_index=7, value=1),
        ]
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_consecutive_small_runs(self):
        """연속된 여러 작은 run이 존재하고, 이들이 병합되지 않는 경우"""
        x = [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]
        sense = 3
        expected = []
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)

    def test_non_consecutive_merges(self):
        """무시된 run을 통해 양 옆 run이 병합되지만, 전체적으로는 분리되는 경우"""
        # 병합 로직이 제거되었으므로, 각 run은 독립적으로 처리됩니다.
        x = [1, 1, 2, 1, 1, 3, 1, 1]
        sense = 2
        expected = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=3, end_index=4, value=1),
            Run(start_index=6, end_index=7, value=1),
        ]
        result = run_length_encoding(x, sense)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
