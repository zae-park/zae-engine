import unittest

import torch
from zae_engine.nn_night.blocks import unet_block


class TestRSUBlock(unittest.TestCase):
    def setUp(self):
        # unet_block.RSUBlock 초기화: ch_in=3, mid_ch=32, ch_out=3, height=7, dilation_height=2
        self.rsu_block = unet_block.RSUBlock(ch_in=3, mid_ch=32, ch_out=3, height=7, dilation_height=2)

    def test_output_shape(self):
        # 입력 텐서 생성
        input_tensor = torch.randn(1, 3, 256, 256)
        # 순전파 수행
        output = self.rsu_block(input_tensor)
        # 출력 텐서의 형태 확인
        self.assertEqual(output.shape, input_tensor.shape, "unet_block.RSUBlock output shape should match input shape.")

    def test_dilation_height(self):
        # unet_block.RSUBlock을 다른 dilation_height로 초기화
        rsu_block_dilate = unet_block.RSUBlock(ch_in=3, mid_ch=32, ch_out=3, height=7, dilation_height=3)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = rsu_block_dilate(input_tensor)
        # 출력 텐서의 형태 확인
        self.assertEqual(
            output.shape,
            input_tensor.shape,
            "unet_block.RSUBlock output shape should match input shape with dilation_height=3.",
        )

    def test_height_variation(self):
        # unet_block.RSUBlock을 다른 height로 초기화 (예: RSU4)
        rsu4 = unet_block.RSUBlock(ch_in=3, mid_ch=32, ch_out=3, height=4, dilation_height=2)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = rsu4(input_tensor)
        # 출력 텐서의 형태 확인
        self.assertEqual(
            output.shape, input_tensor.shape, "unet_block.RSUBlock output shape should match input shape for RSU4."
        )

    def test_channel_change(self):
        # unet_block.RSUBlock을 다른 채널로 초기화 (ch_in=3, mid_ch=64, ch_out=6)
        rsu_block_channel = unet_block.RSUBlock(ch_in=3, mid_ch=64, ch_out=6, height=7, dilation_height=2)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = rsu_block_channel(input_tensor)
        # 출력 채널 확인
        self.assertEqual(output.shape, (1, 6, 256, 256), "unet_block.RSUBlock output channels should be 6.")

    def test_different_input_sizes(self):
        # 다양한 입력 크기에 대해 unet_block.RSUBlock 테스트
        input_sizes = [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 768, 512)]
        for size in input_sizes:
            with self.subTest(size=size):
                input_tensor = torch.randn(size)
                output = self.rsu_block(input_tensor)
                self.assertEqual(
                    output.shape,
                    size,
                    f"unet_block.RSUBlock output shape should match input shape for input size {size}.",
                )

    def test_forward_pass(self):
        # unet_block.RSUBlock의 순전파가 정상적으로 동작하는지 확인
        input_tensor = torch.randn(1, 3, 256, 256)
        try:
            output = self.rsu_block(input_tensor)
            # Forward pass가 예외 없이 완료되었는지 확인
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"unet_block.RSUBlock forward pass failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
