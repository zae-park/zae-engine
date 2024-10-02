import unittest

from zae_engine.operation.run_length import RunLengthCodec, Run, RunList


class TestRunLengthCodec(unittest.TestCase):
    def setUp(self):
        self.codec = RunLengthCodec()

    def test_empty_list(self):
        """Test encoding and decoding of an empty list."""
        x = []
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        self.assertEqual(encoded_runs.raw(), [])
        self.assertEqual(encoded_runs.filtered(), [])
        decoded = self.codec.decode(encoded_runs)
        self.assertEqual(decoded, [])

    def test_single_run(self):
        """Test encoding and decoding of a single run."""
        x = [1, 1, 1, 1]
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [Run(start_index=0, end_index=3, value=1)]
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_raw)
        decoded = self.codec.decode(encoded_runs)
        self.assertEqual(decoded, x)

    def test_multiple_runs(self):
        """Test encoding and decoding of multiple runs."""
        x = [1, 1, 2, 2, 2, 3, 3, 1, 1, 4]
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=4, value=2),
            Run(start_index=5, end_index=6, value=3),
            Run(start_index=7, end_index=8, value=1),
            Run(start_index=9, end_index=9, value=4),
        ]
        expected_filtered = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=4, value=2),
            Run(start_index=5, end_index=6, value=3),
            Run(start_index=7, end_index=8, value=1),
        ]
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        decoded_raw = self.codec.decode(encoded_runs)
        self.assertEqual(decoded_raw, x)

    def test_runs_below_sense(self):
        """Test encoding and decoding when some runs are below the sense threshold."""
        x = [1, 1, 2, 3, 1, 1]
        sense = 3
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=2, value=2),
            Run(start_index=3, end_index=3, value=3),
            Run(start_index=4, end_index=5, value=1),
        ]
        expected_filtered = []
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        # Decode raw runs
        decoded_raw = self.codec.decode(encoded_runs)
        expected_decoded_raw = [1, 1, 2, 3, 1, 1]
        self.assertEqual(decoded_raw, expected_decoded_raw)
        # Decode filtered runs
        filtered_encoded_runs = RunList(all_runs=encoded_runs.filtered(), sense=sense, original_length=6)
        decoded_filtered = self.codec.decode(filtered_encoded_runs)
        expected_decoded_filtered = [0, 0, 0, 0, 0, 0]
        self.assertEqual(decoded_filtered, expected_decoded_filtered)

    def test_all_runs_below_sense(self):
        """Test encoding and decoding when all runs are below the sense threshold."""
        x = [1, 1, 2, 2]
        sense = 3
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [Run(start_index=0, end_index=1, value=1), Run(start_index=2, end_index=3, value=2)]
        expected_filtered = []
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        # Decode raw runs
        decoded_raw = self.codec.decode(encoded_runs)
        expected_decoded_raw = x
        self.assertEqual(decoded_raw, expected_decoded_raw)
        # Decode filtered runs
        filtered_encoded_runs = RunList(all_runs=encoded_runs.filtered(), sense=sense, original_length=4)
        decoded_filtered = self.codec.decode(filtered_encoded_runs)
        expected_decoded_filtered = [0, 0, 0, 0]
        self.assertEqual(decoded_filtered, expected_decoded_filtered)

    def test_decode_with_filtered_runs(self):
        """Test decoding using only filtered runs."""
        x = [1, 1, 2, 1, 1]
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=2, value=2),
            Run(start_index=3, end_index=4, value=1),
        ]
        expected_filtered = [Run(start_index=0, end_index=1, value=1), Run(start_index=3, end_index=4, value=1)]
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        # Decode filtered runs
        filtered_encoded_runs = RunList(all_runs=encoded_runs.filtered(), sense=sense, original_length=5)
        decoded_filtered = self.codec.decode(filtered_encoded_runs)
        expected_decoded_filtered = [1, 1, 0, 1, 1]
        self.assertEqual(decoded_filtered, expected_decoded_filtered)

    def test_decode_with_large_length(self):
        """Test decoding with a specified length larger than the maximum end_index."""
        x = [1, 1, 2, 2, 2]
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [Run(start_index=0, end_index=1, value=1), Run(start_index=2, end_index=4, value=2)]
        expected_filtered = expected_raw  # All runs meet sense
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        # Decode with larger length by using original_length
        decoded = self.codec.decode(encoded_runs)
        expected_decoded = [1, 1, 2, 2, 2]
        self.assertEqual(decoded, expected_decoded)

    def test_encode_with_single_element_runs(self):
        """Test encoding and decoding when runs have single elements."""
        x = [1, 2, 3, 4]
        sense = 1
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [
            Run(start_index=0, end_index=0, value=1),
            Run(start_index=1, end_index=1, value=2),
            Run(start_index=2, end_index=2, value=3),
            Run(start_index=3, end_index=3, value=4),
        ]
        expected_filtered = expected_raw  # All runs meet sense=1
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        decoded = self.codec.decode(encoded_runs)
        self.assertEqual(decoded, x)

    def test_encode_with_varied_run_lengths(self):
        """Test encoding and decoding with runs of varied lengths."""
        x = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [
            Run(start_index=0, end_index=2, value=1),
            Run(start_index=3, end_index=4, value=2),
            Run(start_index=5, end_index=8, value=3),
        ]
        expected_filtered = expected_raw  # All runs meet sense=2
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        decoded = self.codec.decode(encoded_runs)
        self.assertEqual(decoded, x)

    def test_encode_with_runs_below_sense(self):
        """Test encoding with some runs below the sense threshold."""
        x = [1, 1, 2, 2, 3, 1, 1]
        sense = 3
        encoded_runs = self.codec.encode(x, sense)
        expected_raw = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=3, value=2),
            Run(start_index=4, end_index=4, value=3),
            Run(start_index=5, end_index=6, value=1),
        ]
        expected_filtered = []
        self.assertEqual(encoded_runs.raw(), expected_raw)
        self.assertEqual(encoded_runs.filtered(), expected_filtered)
        # Decode filtered runs
        filtered_encoded_runs = RunList(all_runs=encoded_runs.filtered(), sense=sense, original_length=7)
        decoded_filtered = self.codec.decode(filtered_encoded_runs)
        expected_decoded_filtered = [0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(decoded_filtered, expected_decoded_filtered)

    def test_call_encode(self):
        """Test the __call__ method for encoding."""
        x = [1, 1, 2, 2, 2, 3, 3, 1, 1, 4]
        sense = 2
        encoded_runs = self.codec(x, sense=sense)
        expected_filtered = [
            Run(start_index=0, end_index=1, value=1),
            Run(start_index=2, end_index=4, value=2),
            Run(start_index=5, end_index=6, value=3),
            Run(start_index=7, end_index=8, value=1),
        ]
        self.assertEqual(encoded_runs.filtered(), expected_filtered)

    def test_call_decode(self):
        """Test the __call__ method for decoding."""
        x = [1, 1, 2, 2, 2, 3, 3, 1, 1, 4]
        sense = 2
        encoded_runs = self.codec.encode(x, sense)
        decoded = self.codec(encoded_runs)
        self.assertEqual(decoded, x)


# 테스트 실행
if __name__ == "__main__":
    unittest.main()
