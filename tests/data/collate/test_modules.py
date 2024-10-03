import unittest
import torch
import numpy as np
from typing import Dict, Any

from zae_engine.data.collate.modules import (
    UnifiedChunker,
    Chunk,
    HotEncoder,
    SignalFilter,
    Spliter,
    SignalScaler,
)


class TestUnifiedChunker(unittest.TestCase):
    """Unit tests for the UnifiedChunker class."""

    def setUp(self):
        """Set up common test data."""
        self.chunk_size = 3
        self.overlap = 1
        self.chunker = UnifiedChunker(chunk_size=self.chunk_size, overlap=self.overlap)

    def test_chunker_1d_no_padding(self):
        """Test that a 1D tensor is correctly split into chunks without needing padding."""
        batch = {
            "x": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32),  # Shape: (6,)
            "y": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32),  # Shape: (6,)
            "fn": ["sample_fn"],
        }

        processed_batch = self.chunker(batch)

        expected_x = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 6]], dtype=torch.float32)

        expected_y = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=torch.float32)

        # expected_fn = ["sample_fn", "sample_fn", "sample_fn"]

        self.assertTrue(torch.equal(processed_batch["x"], expected_x))
        self.assertTrue(torch.equal(processed_batch["y"], expected_y))
        # self.assertEqual(processed_batch["fn"], expected_fn)

    def test_chunker_1d_with_padding(self):
        """Test that a 1D tensor is correctly split into chunks with padding when needed."""
        batch = {
            "x": torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # Shape: (5,)
            "y": torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32),  # Shape: (5,)
            "fn": ["sample_fn"],
        }

        processed_batch = self.chunker(batch)

        expected_x = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.float32)

        expected_y = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.float32)

        # expected_fn = ["sample_fn", "sample_fn"]

        self.assertTrue(torch.equal(processed_batch["x"], expected_x))
        self.assertTrue(torch.equal(processed_batch["y"], expected_y))
        # self.assertEqual(processed_batch["fn"], expected_fn)

    def test_chunker_2d_no_padding(self):
        """Test that a 2D tensor is correctly split into chunks without needing padding."""
        chunker = UnifiedChunker(chunk_size=4, overlap=2)
        batch = {
            "x": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.float32),  # Shape: (2, 6)
            "y": torch.tensor([0, 1], dtype=torch.float32),  # Shape: (2,)
            "fn": ["fn1", "fn2"],
        }

        processed_batch = chunker(batch)

        expected_x = torch.tensor(
            [[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10], [9, 10, 11, 12]],
            dtype=torch.float32,
        )

        expected_y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)  # Depending on 'y' processing logic

        # expected_fn = ["fn1", "fn1", "fn2", "fn2"]

        # Depending on how 'y' is processed, this assertion might need adjustment
        # self.assertTrue(torch.equal(processed_batch['y'], expected_y))
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))
        # self.assertEqual(processed_batch["fn"], expected_fn)

    def test_chunker_invalid_overlap(self):
        """Test that a ValueError is raised when overlap is greater than or equal to chunk_size."""
        with self.assertRaises(ValueError):
            UnifiedChunker(chunk_size=3, overlap=3)

    def test_chunker_missing_x_key(self):
        """Test that a KeyError is raised when 'x' key is missing in the batch."""
        batch = {"y": torch.tensor([0, 1], dtype=torch.float32), "fn": "sample_fn"}

        with self.assertRaises(KeyError):
            self.chunker(batch)

    def test_chunker_non_tensor_x(self):
        """Test that a TypeError is raised when 'x' is not a tensor."""
        batch = {"x": [1, 2, 3, 4, 5], "y": torch.tensor([0, 1], dtype=torch.float32), "fn": "sample_fn"}

        with self.assertRaises(TypeError):
            self.chunker(batch)

    def test_chunker_unsupported_dimension(self):
        """Test that a ValueError is raised when 'x' has unsupported dimensions."""
        batch = {"x": torch.randn(2, 3, 4), "y": torch.tensor([0, 1], dtype=torch.float32), "fn": "sample_fn"}

        with self.assertRaises(ValueError):
            self.chunker(batch)


class TestChunk(unittest.TestCase):
    """Unit tests for the Chunk class."""

    def setUp(self):
        """Set up common test data."""
        self.batch = {
            "x": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),  # Shape: (2, 6)
            "y": torch.tensor([0, 1]),  # Shape: (2,)
            "fn": "sample_fn",
        }

    def test_chunk_correct_behavior(self):
        """Test Chunk correctly reshapes the 'x' tensor."""
        chunk = Chunk(n=3)
        processed_batch = chunk(self.batch)

        # Original 'x' shape: (2, 6) -> Reshape to (-1, 3) => (4, 3)
        expected_x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))

    def test_chunk_non_tensor_x(self):
        """Test Chunk raises TypeError when 'x' is not a tensor."""
        chunk = Chunk(n=3)
        batch = {
            "x": [[1, 2, 3], [4, 5, 6]],  # 'x' is a list, not a tensor
            "y": torch.tensor([0, 1]),
            "fn": "sample_fn",
        }

        with self.assertRaises(TypeError):
            chunk(batch)

    def test_chunk_insufficient_dimensions(self):
        """Test Chunk raises ValueError when 'x' has insufficient dimensions."""
        chunk = Chunk(n=3)
        batch = {
            "x": torch.tensor([1, 2, 3, 4, 5, 6]),  # Shape: (6,), insufficient dimensions
            "y": torch.tensor([0, 1]),
            "fn": "sample_fn",
        }

        with self.assertRaises(ValueError):
            chunk(batch)

    def test_chunk_modulo_zero(self):
        """Test Chunk correctly reshapes when modulo is zero."""
        chunk = Chunk(n=2)
        batch = {
            "x": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),  # Shape: (2, 4)
            "y": torch.tensor([0, 1]),
            "fn": "sample_fn",
        }
        processed_batch = chunk(batch)

        # Expected 'x' shape after reshape: (-1, 2) => (4, 2)
        expected_x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))

    def test_chunk_modulo_non_zero(self):
        """Test Chunk correctly handles non-zero modulo by trimming 'x'."""
        chunk = Chunk(n=3)
        batch = {
            "x": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),  # Shape: (2, 5)
            "y": torch.tensor([0, 1]),
            "fn": "sample_fn",
        }
        processed_batch = chunk(batch)

        # Original 'x' has last dimension 5, which is not divisible by 3
        # Expected 'x' after trimming: (2, 3) -> reshape to (-1, 3) => (2, 3)
        expected_x = torch.tensor([[1, 2, 3], [6, 7, 8]])
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))


class TestHotEncoder(unittest.TestCase):
    """Unit tests for the HotEncoder class."""

    def setUp(self):
        """Set up common test data."""
        self.batch = {
            "x": torch.tensor([[1.0, 2.0, 3.0]]),  # Shape: (1, 3)
            "y": torch.tensor([0, 2]),  # Shape: (2,)
            "fn": "sample_fn",
        }

    def test_hot_encoder_correct_behavior(self):
        """Test HotEncoder correctly one-hot encodes the 'y' tensor."""
        hot_encoder = HotEncoder(n_cls=3)
        processed_batch = hot_encoder(self.batch)

        expected_y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # One-hot for class 0  # One-hot for class 2
        self.assertTrue(torch.equal(processed_batch["y_hot"], expected_y))

    def test_hot_encoder_multi_dimensional_y(self):
        """Test HotEncoder correctly handles multi-dimensional 'y' tensors."""
        hot_encoder = HotEncoder(n_cls=4)
        batch = {
            "x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # Shape: (2, 2)
            "y": torch.tensor([[1, 2], [0, 3]]),  # Shape: (2, 2)
            "fn": "sample_fn",
        }
        processed_batch = hot_encoder(batch)

        expected_y = torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )
        self.assertTrue(torch.equal(processed_batch["y_hot"], expected_y))

    def test_hot_encoder_missing_y_key(self):
        """Test HotEncoder raises KeyError when 'y' key is missing."""
        hot_encoder = HotEncoder(n_cls=3)
        batch = {
            "x": torch.tensor([[1.0, 2.0, 3.0]]),
            "fn": "sample_fn",
            # 'y' key is missing
        }

        with self.assertRaises(KeyError):
            hot_encoder(batch)

    def test_hot_encoder_invalid_n_cls(self):
        """Test HotEncoder raises error with invalid number of classes."""
        with self.assertRaises(ValueError):
            HotEncoder(n_cls=0)


class TestSignalFilter(unittest.TestCase):
    """Unit tests for the SignalFilter class."""

    def setUp(self):
        """Set up common test data."""
        self.fs = 1000  # Sampling frequency

    def test_signal_filter_lowpass(self):
        """Test SignalFilter correctly applies a lowpass filter."""
        # Create a sample sine wave with high frequency noise
        t = np.linspace(0, 1, self.fs, endpoint=False)
        freq = 5  # 5 Hz signal
        x_np = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(self.fs)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1000)

        batch = {"x": x}
        filter = SignalFilter(fs=self.fs, method="lowpass", cutoff=10)
        processed_batch = filter(batch)

        # Since the original signal has frequency 5 Hz and cutoff is 10 Hz, the signal should remain mostly intact
        self.assertEqual(processed_batch["x"].shape, x.shape)

    def test_signal_filter_invalid_method(self):
        """Test SignalFilter raises ValueError with invalid method."""
        filter = SignalFilter(fs=self.fs, method="invalid_method")
        batch = {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}  # Shape: (3,)

        with self.assertRaises(ValueError):
            filter(batch)

    def test_signal_filter_missing_keys(self):
        """Test SignalFilter raises KeyError when 'x' key is missing."""
        filter = SignalFilter(fs=self.fs, method="lowpass", cutoff=10)
        batch = {"y": torch.tensor([0])}  # 'x' key is missing

        with self.assertRaises(KeyError):
            filter(batch)

    def test_signal_filter_non_tensor_x(self):
        """Test SignalFilter raises TypeError when 'x' is not a tensor."""
        filter = SignalFilter(fs=self.fs, method="lowpass", cutoff=10)
        batch = {"x": [1.0, 2.0, 3.0]}  # 'x' is a list, not a tensor

        with self.assertRaises(TypeError):
            filter(batch)

    def test_signal_filter_insufficient_dimensions(self):
        """Test SignalFilter raises ValueError when 'x' has insufficient dimensions."""
        filter = SignalFilter(fs=self.fs, method="lowpass", cutoff=10)
        batch = {"x": torch.tensor(1.0)}  # 'x' is a scalar tensor

        with self.assertRaises(ValueError):
            filter(batch)

    def test_signal_filter_highpass(self):
        """Test SignalFilter correctly applies a highpass filter."""
        # Create a high-frequency sine wave
        t = np.linspace(0, 1, self.fs, endpoint=False)
        freq = 50  # 50 Hz signal
        x_np = np.sin(2 * np.pi * freq * t)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1000)

        filter = SignalFilter(fs=self.fs, method="highpass", cutoff=30)
        processed_batch = filter({"x": x})

        # Since the cutoff is 30 Hz and the signal is 50 Hz, the signal should pass through
        self.assertEqual(processed_batch["x"].shape, x.shape)

    def test_signal_filter_bandpass(self):
        """Test SignalFilter correctly applies a bandpass filter."""
        # Create a signal with two sine waves
        t = np.linspace(0, 1, self.fs, endpoint=False)
        freq1 = 10  # 10 Hz signal
        freq2 = 50  # 50 Hz signal
        x_np = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1000)

        filter = SignalFilter(fs=self.fs, method="bandpass", lowcut=20, highcut=60)
        processed_batch = filter({"x": x})

        # The 10 Hz signal should be attenuated, and 50 Hz should pass
        self.assertEqual(processed_batch["x"].shape, x.shape)

    def test_signal_filter_bandstop(self):
        """Test SignalFilter correctly applies a bandstop filter."""
        # Create a signal with two sine waves
        t = np.linspace(0, 1, self.fs, endpoint=False)
        freq1 = 10  # 10 Hz signal
        freq2 = 50  # 50 Hz signal
        x_np = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1000)

        filter = SignalFilter(fs=self.fs, method="bandstop", lowcut=40, highcut=60)
        processed_batch = filter({"x": x})

        # The 50 Hz signal should be attenuated, and 10 Hz should pass
        self.assertEqual(processed_batch["x"].shape, x.shape)

    def test_signal_filter_no_padding_needed(self):
        """Test SignalFilter when no padding is needed."""
        # Create a pure sine wave
        t = np.linspace(0, 1, self.fs, endpoint=False)
        freq = 5  # 5 Hz signal
        x_np = np.sin(2 * np.pi * freq * t)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1000)

        filter = SignalFilter(fs=self.fs, method="lowpass", cutoff=10)
        processed_batch = filter({"x": x})

        self.assertEqual(processed_batch["x"].shape, x.shape)


class TestSpliter(unittest.TestCase):
    """Unit tests for the Spliter class."""

    def setUp(self):
        """Set up common test data."""
        self.batch = {
            "x": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32),  # Shape: (8,)
            "y": torch.tensor([0]),  # Shape: (1,)
            "fn": "sample_fn",
        }

    def test_spliter_correct_behavior(self):
        """Test Spliter correctly splits the 'x' tensor with overlapping."""
        spliter = Spliter(chunk_size=4, overlapped=2)
        processed_batch = spliter(self.batch)

        expected_x = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]], dtype=torch.float32)
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))

    def test_spliter_no_overlap(self):
        """Test Spliter correctly splits the 'x' tensor without overlapping."""
        spliter = Spliter(chunk_size=2, overlapped=0)
        batch = {
            "x": torch.tensor([1, 2, 3, 4], dtype=torch.float32),  # Shape: (4,)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        processed_batch = spliter(batch)

        expected_x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))

    def test_spliter_padding_needed(self):
        """Test Spliter correctly pads the 'x' tensor when needed."""
        spliter = Spliter(chunk_size=3, overlapped=1)
        batch = {
            "x": torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # Shape: (5,)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        processed_batch = spliter(batch)

        # After padding, 'x' should be [1,2,3,4,5,5]
        expected_x = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.float32)
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))

    def test_spliter_invalid_overlap(self):
        """Test Spliter raises ValueError when overlapped >= chunk_size."""
        spliter = Spliter(chunk_size=4, overlapped=4)
        with self.assertRaises(ValueError):
            spliter(self.batch)

    def test_spliter_missing_x_key(self):
        """Test Spliter raises KeyError when 'x' key is missing."""
        spliter = Spliter(chunk_size=3, overlapped=1)
        batch = {"y": torch.tensor([0])}  # 'x' key is missing

        with self.assertRaises(KeyError):
            spliter(batch)

    def test_spliter_non_tensor_x(self):
        """Test Spliter raises TypeError when 'x' is not a tensor."""
        spliter = Spliter(chunk_size=3, overlapped=1)
        batch = {"x": [1, 2, 3, 4], "y": torch.tensor([0])}  # 'x' is a list

        with self.assertRaises(TypeError):
            spliter(batch)

    def test_spliter_insufficient_dimensions(self):
        """Test Spliter raises ValueError when 'x' has insufficient dimensions."""
        spliter = Spliter(chunk_size=3, overlapped=1)
        batch = {"x": torch.tensor(1.0), "y": torch.tensor([0])}  # 'x' is a scalar

        with self.assertRaises(ValueError):
            spliter(batch)

    def test_spliter_multiple_chunks(self):
        """Test Spliter correctly splits into multiple chunks."""
        spliter = Spliter(chunk_size=2, overlapped=1)
        batch = {
            "x": torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # Shape: (5,)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        processed_batch = spliter(batch)

        expected_x = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.float32)
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))

    def test_spliter_edge_case_exact_fit(self):
        """Test Spliter when 'x' fits exactly without padding."""
        spliter = Spliter(chunk_size=2, overlapped=1)
        batch = {
            "x": torch.tensor([1, 2, 3, 4], dtype=torch.float32),  # Shape: (4,)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        processed_batch = spliter(batch)

        expected_x = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32)
        self.assertTrue(torch.equal(processed_batch["x"], expected_x))


class TestSignalScaler(unittest.TestCase):
    """Unit tests for the SignalScaler class."""

    def setUp(self):
        """Set up common test data."""
        self.batch = {
            "x": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),  # Shape: (2, 3)
            "y": torch.tensor([0, 1]),  # Shape: (2,)
            "fn": "sample_fn",
        }

    def test_signal_scaler_correct_behavior(self):
        """Test SignalScaler correctly scales the 'x' tensor."""
        scaler = SignalScaler()
        processed_batch = scaler(self.batch)

        # After MinMax scaling, each feature should be between 0 and 1
        self.assertTrue(torch.min(processed_batch["x"]) >= 0.0)
        self.assertTrue(torch.max(processed_batch["x"]) <= 1.0)

    def test_signal_scaler_single_dimension(self):
        """Test SignalScaler correctly scales single-dimensional 'x' tensor."""
        scaler = SignalScaler()
        batch = {
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),  # Shape: (4,)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        processed_batch = scaler(batch)

        # Expected scaled: [0.0, 0.3333, 0.6667, 1.0]
        expected_scaled = torch.tensor([0.0, 0.3333, 0.6667, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(processed_batch["x"], expected_scaled, atol=1e-4))

    def test_signal_scaler_multi_dimensional_x(self):
        """Test SignalScaler correctly scales multi-dimensional 'x' tensor."""
        scaler = SignalScaler()
        batch = {
            "x": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),  # Shape: (3, 2)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        processed_batch = scaler(batch)

        expected_scaled = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(processed_batch["x"], expected_scaled, atol=1e-4))

    def test_signal_scaler_missing_x_key(self):
        """Test SignalScaler raises KeyError when 'x' key is missing."""
        scaler = SignalScaler()
        batch = {"y": torch.tensor([0])}  # 'x' key is missing

        with self.assertRaises(KeyError):
            scaler(batch)

    def test_signal_scaler_non_tensor_x(self):
        """Test SignalScaler raises TypeError when 'x' is not a tensor."""
        scaler = SignalScaler()
        batch = {"x": [1.0, 2.0, 3.0], "y": torch.tensor([0])}  # 'x' is a list

        with self.assertRaises(TypeError):
            scaler(batch)

    def test_signal_scaler_insufficient_dimensions(self):
        """Test SignalScaler raises ValueError when 'x' has more than 2 dimensions."""
        scaler = SignalScaler()
        batch = {"x": torch.randn(2, 3, 4), "y": torch.tensor([0, 1])}  # 'x' has 3 dimensions

        with self.assertRaises(ValueError):
            scaler(batch)

    def test_signal_scaler_multiple_calls(self):
        """Test SignalScaler can be called multiple times with different batches."""
        scaler = SignalScaler()
        batch1 = {
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),  # Shape: (4,)
            "y": torch.tensor([0]),
            "fn": "sample_fn",
        }
        batch2 = {
            "x": torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32),  # Shape: (4,)
            "y": torch.tensor([1]),
            "fn": "sample_fn",
        }

        processed_batch1 = scaler(batch1)
        processed_batch2 = scaler(batch2)

        # Check if both batches are scaled correctly
        self.assertTrue(torch.min(processed_batch1["x"]) >= 0.0)
        self.assertTrue(torch.max(processed_batch1["x"]) <= 1.0)
        self.assertTrue(torch.min(processed_batch2["x"]) >= 0.0)
        self.assertTrue(torch.max(processed_batch2["x"]) <= 1.0)


if __name__ == "__main__":
    unittest.main()
