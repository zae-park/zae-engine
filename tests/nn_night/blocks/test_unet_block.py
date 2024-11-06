import unittest
import torch

from zae_engine.nn_night.blocks.unet_block import RSUBlock


class TestRSUBlock(unittest.TestCase):
    def setUp(self):
        # 기본 설정
        self.ch_in = 3
        self.ch_mid = 32
        self.ch_out = 64
        self.height = 7
        self.dilation_height = 7
        self.pool_size = 2
        self.model = RSUBlock(
            ch_in=self.ch_in,
            ch_mid=self.ch_mid,
            ch_out=self.ch_out,
            height=self.height,
            dilation_height=self.dilation_height,
            pool_size=self.pool_size,
        )

    def test_initialization(self):
        """RSUBlock이 올바르게 초기화되었는지 확인합니다."""
        self.assertEqual(len(self.model.encoder_blocks), self.height - 1, "Incorrect number of encoder blocks.")
        self.assertEqual(len(self.model.pools), self.height - 1, "Incorrect number of pooling layers.")
        self.assertEqual(len(self.model.decoder_blocks), self.height - 1, "Incorrect number of decoder blocks.")
        self.assertEqual(len(self.model.ups), self.height - 1, "Incorrect number of upsampling layers.")

    def test_forward_pass_default_config(self):
        """기본 설정에서 순전파가 정상적으로 동작하고 출력 형태가 올바른지 확인합니다."""
        x = torch.randn(1, self.ch_in, 256, 256)
        try:
            output = self.model(x)
            expected_shape = (1, self.ch_out, 256, 256)
            self.assertEqual(output.shape, expected_shape, "Output shape mismatch.")
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

    def test_forward_pass_different_input_sizes(self):
        """다양한 입력 크기에서 순전파가 정상적으로 동작하고 출력 형태가 올바른지 확인합니다."""
        input_sizes = [(1, self.ch_in, 128, 128), (2, self.ch_in, 256, 256), (4, self.ch_in, 512, 512)]
        for size in input_sizes:
            with self.subTest(size=size):
                x = torch.randn(*size)
                try:
                    output = self.model(x)
                    expected_shape = (size[0], self.ch_out, size[2], size[3])
                    self.assertEqual(output.shape, expected_shape, f"Output shape mismatch for input size {size}.")
                except Exception as e:
                    self.fail(f"Forward pass failed for input size {size} with exception: {e}")

    def test_forward_pass_custom_channels(self):
        """입력 및 출력 채널을 변경하여 순전파가 정상적으로 동작하는지 확인합니다."""
        custom_ch_in = 1
        custom_ch_mid = 16
        custom_ch_out = 32
        model = RSUBlock(
            ch_in=custom_ch_in,
            ch_mid=custom_ch_mid,
            ch_out=custom_ch_out,
            height=self.height,
            dilation_height=self.dilation_height,
            pool_size=self.pool_size,
        )
        model.eval()
        x = torch.randn(1, custom_ch_in, 256, 256)
        try:
            output = model(x)
            expected_shape = (1, custom_ch_out, 256, 256)
            self.assertEqual(output.shape, expected_shape, "Output shape mismatch with custom channels.")
        except Exception as e:
            self.fail(f"Forward pass with custom channels failed with exception: {e}")

    def test_forward_pass_dilation_mode(self):
        """Dilation 모드에서 순전파가 정상적으로 동작하는지 확인합니다."""
        # height=4, dilation_height=2 설정으로 이미 dilation mode 적용됨
        x = torch.randn(1, self.ch_in, 256, 256)
        try:
            output = self.model(x)
            expected_shape = (1, self.ch_out, 256, 256)
            self.assertEqual(output.shape, expected_shape, "Output shape mismatch in dilation mode.")
        except Exception as e:
            self.fail(f"Forward pass in dilation mode failed with exception: {e}")

    def test_model_parameters_exist(self):
        """모델에 최소한 하나 이상의 파라미터가 존재하는지 확인합니다."""
        self.assertTrue(len(list(self.model.parameters())) > 0, "RSUBlock has no parameters.")


if __name__ == "__main__":
    unittest.main()
