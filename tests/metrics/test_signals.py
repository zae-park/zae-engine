import unittest

import numpy as np

import unittest
import numpy as np
import torch
from zae_engine.metrics.signals import rms, mse, signal_to_noise, peak_signal_to_noise
from zae_engine.utils.io import example_ecg


class TestMetrics(unittest.TestCase):

    def test_rms(self):
        signal_np = np.array([1, 2, 3, 4, 5])
        signal_torch = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

        expected_rms = 3.3166247903554

        self.assertAlmostEqual(rms(signal_np), expected_rms, places=6)
        self.assertAlmostEqual(rms(signal_torch).item(), expected_rms, places=6)

    def test_mse(self):
        signal1_np = np.array([1, 2, 3, 4, 5])
        signal2_np = np.array([1, 2, 3, 4, 6])
        signal1_torch = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        signal2_torch = torch.tensor([1, 2, 3, 4, 6], dtype=torch.float32)

        expected_mse = 0.2

        self.assertAlmostEqual(mse(signal1_np, signal2_np).item(), expected_mse, places=6)
        self.assertAlmostEqual(mse(signal1_torch, signal2_torch).item(), expected_mse, places=6)

    def test_signal_to_noise(self):
        signal_np = example_ecg()[0]
        noise_np = np.random.normal(size=len(signal_np))
        signal_torch = torch.tensor(signal_np, dtype=torch.float32)
        noise_torch = torch.tensor(noise_np, dtype=torch.float32)

        # expected_snr = 20.0

        self.assertAlmostEqual(
            signal_to_noise(signal_np, noise_np).item(), signal_to_noise(signal_torch, noise_torch).item(), places=6
        )

    def test_peak_signal_to_noise(self):
        signal_np = np.array([1, 2, 3, 4, 5])
        noise_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        signal_torch = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        noise_torch = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

        expected_psnr = 24.0824

        self.assertAlmostEqual(peak_signal_to_noise(signal_np, noise_np).item(), expected_psnr, places=4)
        self.assertAlmostEqual(peak_signal_to_noise(signal_torch, noise_torch).item(), expected_psnr, places=4)

    # def test_iec(self):
    #     d0 = {"sample": [100, 200, 300, 400, 500], "rhythm": ["(N", "(N", "(AF", "(AF", "(N"]}  # 300~500
    #     d1 = {"sample": [100, 200, 300, 400, 500], "rhythm": ["(AF", "(N", "(AF", "(AF", "(N"]}  # 100~200, 300~500
    #     d2 = {"sample": [100, 200, 300, 400], "rhythm": ["(AF", "(AF", "(AF", "(AF"]}  # 100~
    #     d3 = {"sample": [300, 500, 300, 400], "rhythm": ["(AF", "(AF", "(AF", "(AF"]}  # not sorted
    #     d4 = {"sample": [600, 700, 800, 900], "rhythm": ["(N", "(N", "(N", "(AF"]}  # 900~
    #     self.assertEqual(_measure.iec_60601(d0, d1, 1000, "(AF")[0], _measure.iec_60601(d1, d0, 1000, "(AF")[1])
    #     with self.assertRaises(AssertionError) as err:
    #         _measure.iec_60601(d2, d3, 1000, "(AF")
    #         self.assertTrue(isinstance(err.exception, AssertionError))
    #     with self.assertRaises(KeyError) as err:
    #         _measure.iec_60601(d2, {"a": [1, 2, 3], "b": [5, 6, 7]}, 1000, "(AF")
    #         self.assertTrue(isinstance(err.exception, KeyError))
    #     self.assertEqual(_measure.iec_60601(d2, d4, 800, "(NN")[0], 0)
    #
    # def test_cpsc(self):
    #     true = [10, 15, 30, 52, 69, 180]
    #     predict = [9, 16, 36, 52, 90, 140]
    #     predict2 = [10, 16, 36, 52, 90, 140]
    #     predict3 = [6, 16, 36, 52, 90, 140]
    #     self.assertEqual(_measure.cpsc2021(true, predict2), _measure.cpsc2021(true, predict))
    #     self.assertGreater(_measure.cpsc2021(true, predict2), _measure.cpsc2021(true, predict3))


if __name__ == "__main__":
    unittest.main()
