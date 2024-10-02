import unittest
import numpy as np

from zae_engine.metrics import BijectiveMetrix


class TestBijectiveMetrix(unittest.TestCase):
    def test_normal_case(self):
        """Test the metric calculation with normal input."""
        prediction = np.array([0, 1, 1, 0, 2, 2, 0, 0, 3, 3])
        label = np.array([0, 1, 1, 0, 0, 2, 2, 3, 3, 0])
        num_classes = 3
        metric = BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)
        # Assert that metrics are calculated without errors
        self.assertIsNotNone(metric.bijective_f1)
        self.assertIsNotNone(metric.injective_f1)
        self.assertIsNotNone(metric.surjective_f1)
        # Check that confusion matrices have correct shape
        self.assertEqual(metric.bijective_mat.shape, (num_classes + 1, num_classes + 1))
        self.assertEqual(metric.injective_mat.shape, (num_classes + 1, num_classes + 1))
        self.assertEqual(metric.surjective_mat.shape, (num_classes + 1, num_classes + 1))

    def test_empty_input(self):
        """Test the metric calculation with empty input."""
        prediction = np.array([])
        label = np.array([])
        num_classes = 3
        with self.assertRaises(AssertionError):
            BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)

    def test_mismatched_shapes(self):
        """Test the metric calculation with mismatched input shapes."""
        prediction = np.array([0, 1, 1, 0])
        label = np.array([0, 1, 1])
        num_classes = 3
        with self.assertRaises(AssertionError):
            BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)

    def test_insufficient_num_classes(self):
        """Test the metric calculation with insufficient number of classes."""
        prediction = np.array([0, 1, 1, 0, 2, 2, 0])
        label = np.array([0, 1, 1, 0, 2, 2, 0])
        num_classes = 2  # Should be at least 2 to include classes 0, 1, 2
        metric = BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)
        # Confusion matrix should have shape (num_classes + 1, num_classes + 1)
        self.assertEqual(metric.bijective_mat.shape, (num_classes + 1, num_classes + 1))

    def test_no_runs_detected(self):
        """Test the metric calculation when no runs are detected."""
        prediction = np.zeros(10, dtype=int)
        label = np.zeros(10, dtype=int)
        num_classes = 1
        metric = BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)
        # All metrics should be valid but zero
        self.assertEqual(metric.bijective_count, 0)
        self.assertEqual(metric.injective_count, 0)
        self.assertEqual(metric.surjective_count, 0)
        self.assertEqual(metric.bijective_f1, 0)
        self.assertEqual(metric.injective_f1, 0)
        self.assertEqual(metric.surjective_f1, 0)
        self.assertTrue(np.all(metric.bijective_mat == 0))
        self.assertTrue(np.all(metric.injective_mat == 0))
        self.assertTrue(np.all(metric.surjective_mat == 0))

    def test_different_classes(self):
        """Test the metric calculation with different classes in prediction and label."""
        prediction = np.array([0, 1, 1, 0, 3, 3, 0])
        label = np.array([0, 2, 2, 0, 4, 4, 0])
        num_classes = 4
        metric = BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)
        # Check that confusion matrices have correct shape
        self.assertEqual(metric.bijective_mat.shape, (num_classes + 1, num_classes + 1))

    def test_single_class(self):
        """Test the metric calculation with single class."""
        prediction = np.ones(10, dtype=int)
        label = np.ones(10, dtype=int)
        num_classes = 1
        metric = BijectiveMetrix(prediction, label, num_classes=num_classes, th_run_length=2)
        self.assertEqual(metric.bijective_f1, 1.0)
        self.assertEqual(metric.injective_f1, 1.0)
        self.assertEqual(metric.surjective_f1, 1.0)


if __name__ == "__main__":
    unittest.main()
