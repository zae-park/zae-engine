import unittest
import numpy as np
import torch

from zae_engine.data import HotEncoder, Chunker, SignalFilter


class TestPreprocessingFunctions(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests
        self.hot_encoder = HotEncoder(n_cls=3)
        self.chunker = Chunker(n=2, th=0.5)
        self.signal_filter_bandpass = SignalFilter(fs=100.0, method="bandpass", lowcut=0.5, highcut=50.0)
        self.signal_filter_bandstop = SignalFilter(fs=100.0, method="bandstop", lowcut=59.9, highcut=60.1)
        self.signal_filter_lowpass = SignalFilter(fs=100.0, method="lowpass", cutoff=50.0)
        self.signal_filter_highpass = SignalFilter(fs=100.0, method="highpass", cutoff=0.5)

    def test_hot_encoder(self):
        batch = (torch.tensor([[1, 2, 3]]), torch.tensor([0, 1, 2]), ["file1"])
        x, y, fn = self.hot_encoder(batch)
        expected_y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(y, expected_y)
        self.assertEqual(fn, ["file1"])

    def test_chunker(self):
        batch = (torch.tensor([1, 2, 3]), torch.tensor([0.1, 0.6]), "file1")
        x, y, fn = self.chunker(batch)
        expected_x = torch.tensor([[1, 2, 3], [1, 2, 3]])
        expected_y = torch.tensor([0, 1])
        expected_fn = ["file1", "file1"]
        torch.testing.assert_allclose(x, expected_x)
        torch.testing.assert_allclose(y, expected_y)
        self.assertEqual(fn, expected_fn)

    def test_signal_filter_bandpass(self):
        batch = {"x": torch.tensor(np.random.rand(1, 1000))}
        filtered_batch = self.signal_filter_bandpass(batch)
        self.assertEqual(filtered_batch["x"].shape, torch.Size([1, 1000]))

    def test_signal_filter_bandstop(self):
        batch = {"x": torch.tensor(np.random.rand(1, 1000))}
        filtered_batch = self.signal_filter_bandstop(batch)
        self.assertEqual(filtered_batch["x"].shape, torch.Size([1, 1000]))

    def test_signal_filter_lowpass(self):
        batch = {"x": torch.tensor(np.random.rand(1, 1000))}
        filtered_batch = self.signal_filter_lowpass(batch)
        self.assertEqual(filtered_batch["x"].shape, torch.Size([1, 1000]))

    def test_signal_filter_highpass(self):
        batch = {"x": torch.tensor(np.random.rand(1, 1000))}
        filtered_batch = self.signal_filter_highpass(batch)
        self.assertEqual(filtered_batch["x"].shape, torch.Size([1, 1000]))

    def test_signal_filter_invalid_method(self):
        with self.assertRaises(ValueError):
            SignalFilter(fs=100.0, method="invalid")

    def test_signal_filter_missing_params(self):
        with self.assertRaises(ValueError):
            SignalFilter(fs=100.0, method="bandpass")
        with self.assertRaises(ValueError):
            SignalFilter(fs=100.0, method="bandstop")
        with self.assertRaises(ValueError):
            SignalFilter(fs=100.0, method="lowpass")
        with self.assertRaises(ValueError):
            SignalFilter(fs=100.0, method="highpass")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
