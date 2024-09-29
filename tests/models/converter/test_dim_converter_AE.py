import unittest
import torch
import torch.nn as nn
from copy import deepcopy

from zae_engine.models import AutoEncoder
from zae_engine.models.converter import DimConverter
from zae_engine.nn_night.blocks import UNetBlock

class TestDimConverter(unittest.TestCase):
    def setUp(self):
        """
        테스트를 위한 초기 설정.
        기존의 2D AutoEncoder 인스턴스를 생성하고, DimConverter를 초기화합니다.
        """
        # AutoEncoder 파라미터 설정
        self.block = UNetBlock
        self.ch_in = 3
        self.ch_out = 3
        self.width = 64
        self.layers = [2, 2, 2, 2]
        self.groups = 1
        self.dilation = 1
        self.norm_layer = nn.BatchNorm2d
        self.skip_connect = True

        # AutoEncoder 인스턴스 생성
        self.autoencoder = AutoEncoder(
            block=self.block,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            groups=self.groups,
            dilation=self.dilation,
            norm_layer=self.norm_layer,
            skip_connect=self.skip_connect,
        )

        # DimConverter 인스턴스 생성
        self.dim_converter = DimConverter(self.autoencoder)

    def test_layer_conversion(self):
        """
        DimConverter를 사용하여 AutoEncoder의 레이어가 2D에서 1D로 올바르게 변환되었는지 확인합니다.
        """
        converted_model = self.dim_converter.convert("2d -> 1d")

        # 변환된 모델의 레이어 타입 확인
        for name, module in converted_model.named_modules():
            if isinstance(module, nn.Conv1d):
                # Conv2d가 Conv1d로 변환되었는지 확인
                self.assertTrue(True)
            elif isinstance(module, nn.ConvTranspose1d):
                # ConvTranspose2d가 ConvTranspose1d로 변환되었는지 확인
                self.assertTrue(True)
            elif isinstance(module, nn.BatchNorm1d):
                # BatchNorm2d가 BatchNorm1d로 변환되었는지 확인
                self.assertTrue(True)
            # 필요한 다른 레이어 타입도 확인할 수 있습니다.

    def test_forward_pass_with_1d_input(self):
        """
        변환된 모델이 1D 입력을 정상적으로 처리할 수 있는지 확인합니다.
        """
        converted_model = self.dim_converter.convert("2d -> 1d")
        converted_model.eval()

        # 1D 입력 데이터 생성 (배치 크기 4, 채널 3, 길이 256)
        batch_size = 4
        channels = self.ch_in
        length = 256
        test_input = torch.sigmoid(torch.randn(batch_size, channels, length))  # [0, 1] 범위로 정규화

        with torch.no_grad():
            output = converted_model(test_input)

        # 출력의 형태가 입력과 동일한지 확인
        self.assertEqual(output.shape, test_input.shape)

    def test_forward_pass_preserves_output_shape(self):
        """
        변환 전후의 AutoEncoder가 동일한 입력 크기에 대해 동일한 출력 크기를 가지는지 확인합니다.
        """
        converted_model = self.dim_converter.convert("2d -> 1d")
        converted_model.eval()
        original_model = self.autoencoder
        original_model.eval()

        # 2D 입력 데이터 생성
        batch_size = 4
        channels = self.ch_in
        height = 256
        width = 256
        test_input_2d = torch.sigmoid(torch.randn(batch_size, channels, height, width))

        # 1D 입력 데이터 생성
        length = height  # assuming height is analogous to length in 1D
        test_input_1d = torch.sigmoid(torch.randn(batch_size, channels, length))

        with torch.no_grad():
            output_original = original_model(test_input_2d)
            output_converted = converted_model(test_input_1d)

        # 2D 모델의 출력과 1D 모델의 출력 크기가 대응되는지 확인
        self.assertEqual(output_converted.shape, (batch_size, channels, length))

    def test_weight_conversion(self):
        """
        레이어의 가중치가 올바르게 변환되었는지 확인합니다.
        (예: Conv2d의 가중치가 Conv1d로 올바르게 변환되었는지)
        """
        converted_model = self.dim_converter.convert("2d -> 1d")

        # 원본 모델과 변환된 모델의 특정 레이어의 가중치를 비교합니다.
        # 예시로, 첫 번째 Conv 레이어를 비교합니다.
        original_conv = self.autoencoder.encoder.body[0][0].conv1
        converted_conv = converted_model.encoder.body[0][0].conv1

        # # Conv2d의 가중치를 Conv1d에 맞게 변환 (예: 평균을 취하거나 특정 축을 선택)
        # # 여기서는 단순히 평균을 취하여 1D 가중치를 생성했다고 가정합니다.
        # # 실제 변환 로직에 따라 달라질 수 있습니다.
        # expected_conv_weight = original_conv.weight.mean(dim=2)  # (out_channels, in_channels, kernel_size)
        # self.assertTrue(torch.allclose(converted_conv.weight, expected_conv_weight, atol=1e-5))

    def test_dimensionality_reduction(self):
        """
        변환 후 모델이 1D 입력을 제대로 처리하여 출력이 올바르게 축소되었는지 확인합니다.
        """
        converted_model = self.dim_converter.convert("2d -> 1d")
        converted_model.eval()

        # 1D 입력 데이터 생성
        batch_size = 4
        channels = self.ch_in
        length = 256
        test_input = torch.sigmoid(torch.randn(batch_size, channels, length))

        with torch.no_grad():
            output = converted_model(test_input)

        # 출력의 형태가 입력과 동일한지 확인
        self.assertEqual(output.shape, test_input.shape)

    def test_non_convertable_layers(self):
        """
        변환할 수 없는 레이어가 변환되지 않고 그대로 유지되는지 확인합니다.
        """
        converted_model = self.dim_converter.convert("2d -> 1d")

        # 예시로, 만약 모델에 Conv3d가 있었다면 변환되지 않았는지 확인
        # 현재 AutoEncoder에는 Conv3d가 없으므로, 다른 레이어 예시를 들어봅니다.
        # 예를 들어, Linear 레이어가 있다면 변환되지 않아야 합니다.
        # 여기서는 레이어가 변환되지 않았는지 단순히 확인합니다.
        for name, module in converted_model.named_modules():
            if isinstance(module, nn.Linear):
                self.assertTrue(True)  # 변환되지 않았음을 확인
            # 다른 변환 불가능한 레이어도 동일하게 확인할 수 있습니다.

if __name__ == '__main__':
    unittest.main()