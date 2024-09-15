import unittest

import torch

from zae_engine.models.foundations import unet_brain


class TestUnet(unittest.TestCase):
    test_sample = None

    @classmethod
    def setUpClass(cls) -> None:
        # cls.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        pass

    def setUp(self) -> None:
        # self.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        self.test_sample = torch.randn((2, 3, 256, 256))

    def tearDown(self) -> None:
        pass

    def test_unet(self):
        model = unet_brain()
        output = model(self.test_sample)
        expect_output = self.test_sample.mean(1, keepdim=True)

        self.assertEqual(output.shape, expect_output.shape)


if __name__ == "__main__":
    unittest.main()
