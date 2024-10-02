import unittest

from zae_engine.operation.run_length import RunLengthCodec, Run, RunList


class TestRunListRepr(unittest.TestCase):
    def test_runlist_repr(self):
        """Test the __repr__ method of RunList."""
        runs = [Run(0, 2, 1), Run(3, 5, 2)]
        run_list = RunList(all_runs=runs, sense=1, original_length=6)

        self.assertIn("RunList(all_runs=[", repr(run_list))
        self.assertIn("Run(start_index=0, end_index=2, value=1)", repr(run_list))
        self.assertIn("Run(start_index=3, end_index=5, value=2)", repr(run_list))
        self.assertIn("sense=1, original_length=6)", repr(run_list))

    def test_empty_runlist_repr(self):
        """Test the __repr__ method of RunList with empty runs."""
        runs = []
        run_list = RunList(all_runs=runs, sense=1, original_length=0)
        expected_repr = "RunList(all_runs=[], sense=1, original_length=0)"
        self.assertEqual(repr(run_list), expected_repr)


class TestRunLengthCodec(unittest.TestCase):
    def setUp(self):
        # 기본 설정: tol_merge=20, remove_incomplete=False, merge_closed=False
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
        # Decode with original_length
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
        expected_filtered = []  # No runs meet sense=3
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

    def test_batch_encode(self):
        """Test batch encoding with multiple lists."""
        x_batch = [[1, 1, 2, 2, 2], [3, 3, 3, 4, 4], [5, 5, 5, 5, 5]]
        sense = 2
        encoded_batch = self.codec(x_batch, sense=sense)
        expected_filtered = [
            [Run(start_index=0, end_index=1, value=1), Run(start_index=2, end_index=4, value=2)],
            [Run(start_index=0, end_index=2, value=3), Run(start_index=3, end_index=4, value=4)],
            [Run(start_index=0, end_index=4, value=5)],
        ]
        for encoded, expected in zip(encoded_batch, expected_filtered):
            self.assertEqual(encoded.filtered(), expected)

    def test_batch_decode(self):
        """Test batch decoding with multiple RunList objects."""
        x_batch = [[1, 1, 2, 2, 2], [3, 3, 3, 4, 4], [5, 5, 5, 5, 5]]
        sense = 2
        encoded_batch = self.codec.encode(x_batch, sense=sense)
        decoded_batch = self.codec.decode(encoded_batch)
        self.assertEqual(decoded_batch, x_batch)

    def test_call_encode_batch(self):
        """Test the __call__ method for batch encoding."""
        x_batch = [[1, 1, 2, 2, 2], [3, 3, 3, 4, 4], [5, 5, 5, 5, 5]]
        sense = 2
        encoded_batch = self.codec(x_batch, sense=sense)
        expected_filtered = [
            [Run(start_index=0, end_index=1, value=1), Run(start_index=2, end_index=4, value=2)],
            [Run(start_index=0, end_index=2, value=3), Run(start_index=3, end_index=4, value=4)],
            [Run(start_index=0, end_index=4, value=5)],
        ]
        for encoded, expected in zip(encoded_batch, expected_filtered):
            self.assertEqual(encoded.filtered(), expected)

    def test_call_decode_batch(self):
        """Test the __call__ method for batch decoding."""
        x_batch = [[1, 1, 2, 2, 2], [3, 3, 3, 4, 4], [5, 5, 5, 5, 5]]
        sense = 2
        encoded_batch = self.codec.encode(x_batch, sense=sense)
        decoded_batch = self.codec.decode(encoded_batch)
        self.assertEqual(decoded_batch, x_batch)

    def test_encode_and_decode(self):
        x = [1, 1, 2, 2, 2, 3, 3, 1, 1, 4]
        sense = 2
        run_list = self.codec.encode(x, sense)
        decoded = self.codec.decode(run_list)
        self.assertEqual(decoded, x)

    def test_sanitize_remove_incomplete(self):
        # Runs with incomplete runs (start_index == 0 or end_index == original_length -1)
        runs = [Run(start_index=0, end_index=1, value=1), Run(2, 4, 2), Run(5, 6, 3), Run(7, 9, 1)]
        run_list = RunList(all_runs=runs, sense=2, original_length=10)
        self.codec.remove_incomplete = True
        self.codec.merge_closed = False
        sanitized = self.codec.sanitize(run_list)
        expected = [Run(2, 4, 2), Run(5, 6, 3)]
        self.assertEqual(sanitized.all_runs, expected)

    def test_sanitize_merge_closed(self):
        # Runs that should be merged based on tol_merge
        runs = [Run(2, 4, 2), Run(5, 6, 2), Run(7, 8, 2)]
        run_list = RunList(all_runs=runs, sense=2, original_length=9)
        self.codec.merge_closed = True
        self.codec.remove_incomplete = False
        sanitized = self.codec.sanitize(run_list)
        expected = [Run(2, 8, 2)]  # All runs have same value and are close
        self.assertEqual(sanitized.all_runs, expected)

    def test_sanitize_no_merge(self):
        # Runs that should not be merged because they have different values
        runs = [Run(0, 2, 1), Run(5, 7, 2)]
        run_list = RunList(all_runs=runs, sense=2, original_length=8)
        codec = RunLengthCodec(tol_merge=1, remove_incomplete=False, merge_closed=True)
        sanitized = codec.sanitize(run_list)
        expected = [Run(0, 2, 1), Run(5, 7, 2)]
        self.assertEqual(sanitized.all_runs, expected)

    def test_sanitize_remove_incomplete_and_merge_closed(self):
        # Runs with incomplete runs and runs that should be merged
        runs = [Run(0, 1, 1), Run(2, 4, 2), Run(5, 7, 2), Run(8, 9, 2)]
        run_list = RunList(all_runs=runs, sense=2, original_length=10)
        self.codec.remove_incomplete = True
        self.codec.merge_closed = True
        sanitized = self.codec.sanitize(run_list)
        expected = [Run(2, 7, 2)]  # Runs 2-4 and 5-7 merged; Run ending at 9 removed
        self.assertEqual(expected, sanitized.all_runs)

    def test_no_remove_no_merge(self):
        # Runs without removing incomplete and without merging
        runs = [Run(2, 4, 2), Run(5, 6, 3), Run(7, 8, 1)]
        run_list = RunList(all_runs=runs, sense=2, original_length=9)
        self.codec.merge_closed = False
        self.codec.remove_incomplete = False
        sanitized = self.codec.sanitize(run_list)
        expected = [Run(2, 4, 2), Run(5, 6, 3), Run(7, 8, 1)]
        self.assertEqual(sanitized.all_runs, expected)

    def test_encode_with_sanitization(self):
        # Encode and ensure sanitization is applied
        x = [1, 1, 0, 0, 0, 3, 3, 0, 0, 0]
        sense = 1
        # Initialize codec with sanitization options
        codec = RunLengthCodec(tol_merge=1, remove_incomplete=False, merge_closed=True, base_class=0)
        run_list = codec.encode(x, sense)
        expected_runs = [Run(start_index=0, end_index=1, value=1), Run(start_index=5, end_index=6, value=3)]
        self.assertEqual(expected_runs, run_list.all_runs)

    def test_encode_with_merge_closed(self):
        """Test encoding with merging closed runs of the same value."""
        x = [1, 1, 2, 2, 1, 1, 1, 2, 2, 2]
        x2 = [1, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2]
        sense = 2
        codec = RunLengthCodec(tol_merge=1, remove_incomplete=False, merge_closed=True)
        run_list = codec.encode(x, sense)
        expected_runs = [Run(start_index=0, end_index=9, value=1)]
        self.assertEqual(expected_runs, run_list.all_runs)
        run_list2 = codec.encode(x2, sense)
        expected_runs2 = [Run(start_index=0, end_index=2, value=1), Run(start_index=6, end_index=10, value=1)]
        self.assertEqual(expected_runs2, run_list2.all_runs)

    def test_encode_with_remove_incomplete(self):
        """Test encoding with removing incomplete runs."""
        x = [1, 1, 2, 2, 2, 3, 3, 1, 1, 4]
        sense = 2
        codec = RunLengthCodec(tol_merge=20, remove_incomplete=True, merge_closed=False)
        run_list = codec.encode(x, sense)
        expected_runs = [
            Run(start_index=2, end_index=4, value=2),
            Run(start_index=5, end_index=6, value=3),
            Run(start_index=7, end_index=8, value=1),
        ]
        self.assertEqual(run_list.all_runs, expected_runs)


# 테스트 실행
if __name__ == "__main__":
    unittest.main()
