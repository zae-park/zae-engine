import unittest

import torch
from zae_engine.nn_night.blocks import conv_block


class TestConvBlock(unittest.TestCase):
    def setUp(self):
        # 기본 설정: 3 입력 채널, 3 출력 채널, dilation=1
        self.conv_block = conv_block.ConvBlock(ch_in=3, ch_out=3, dilate=1)

    def test_output_shape(self):
        # 입력 텐서 생성
        input_tensor = torch.randn(1, 3, 256, 256)
        # 순전파 수행
        output = self.conv_block(input_tensor)
        # 출력 텐서의 형태 확인
        self.assertEqual(output.shape, input_tensor.shape, "Output shape should match input shape.")

    def test_dilation(self):
        # dilation=2 설정
        conv_block_dilate = conv_block.ConvBlock(ch_in=3, ch_out=3, dilate=2)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = conv_block_dilate(input_tensor)
        # 출력 텐서의 형태 확인
        self.assertEqual(output.shape, input_tensor.shape, "Output shape should match input shape with dilation=2.")

    def test_channel_change(self):
        # 입력 채널=3, 출력 채널=6
        conv_block_channel = conv_block.ConvBlock(ch_in=3, ch_out=6, dilate=1)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = conv_block_channel(input_tensor)
        # 출력 채널 확인
        self.assertEqual(output.shape, (1, 6, 256, 256), "Output channels should be 6.")

    def test_pre_norm(self):
        # pre_norm=True 설정
        conv_block_pre_norm = conv_block.ConvBlock(ch_in=3, ch_out=3, dilate=1, pre_norm=True)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = conv_block_pre_norm(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape, "Output shape should match input shape with pre_norm=True.")


if __name__ == "__main__":
    unittest.main()
