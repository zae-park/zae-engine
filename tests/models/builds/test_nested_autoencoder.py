import unittest

import torch

from zae_engine.models.builds.nested_autoencoder import NestedUNet


class TestNestedUNet(unittest.TestCase):
    def setUp(self):
        # 기본 설정
        self.in_ch = 3
        self.out_ch = 1
        self.width = 32
        self.heights = [7, 6, 5, 4]
        self.dilation_heights = [2, 2, 2, 2]
        self.model = NestedUNet(
            in_ch=self.in_ch,
            out_ch=self.out_ch,
            width=self.width,
            heights=self.heights,
            dilation_heights=self.dilation_heights,
        )
        self.model.eval()

    def test_forward_pass_default_config(self):
        """기본 설정에서 순전파가 정상적으로 동작하고 출력 형태가 올바른지 확인합니다."""
        x = torch.randn(1, self.in_ch, 256, 256)
        try:
            output, *side_outputs = self.model(x)
            self.assertEqual(output.shape, (1, self.out_ch, 256, 256), "Final output shape mismatch.")
            for i, side in enumerate(side_outputs):
                self.assertEqual(side.shape, (1, self.out_ch, 256, 256), f"Side output {i+1} shape mismatch.")
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

    def test_forward_pass_different_input_sizes(self):
        """다양한 입력 크기에서 순전파가 정상적으로 동작하고 출력 형태가 올바른지 확인합니다."""
        input_sizes = [(1, self.in_ch, 128, 128), (2, self.in_ch, 256, 256), (4, self.in_ch, 512, 512)]
        for size in input_sizes:
            with self.subTest(size=size):
                x = torch.randn(*size)
                try:
                    output, *side_outputs = self.model(x)
                    self.assertEqual(
                        output.shape,
                        (size[0], self.out_ch, size[2], size[3]),
                        f"Final output shape mismatch for input size {size}.",
                    )
                    for i, side in enumerate(side_outputs):
                        self.assertEqual(
                            side.shape,
                            (size[0], self.out_ch, size[2], size[3]),
                            f"Side output {i+1} shape mismatch for input size {size}.",
                        )
                except Exception as e:
                    self.fail(f"Forward pass failed for input size {size} with exception: {e}")

    def test_forward_pass_custom_heights_and_dilations(self):
        """커스텀 heights와 dilation_heights를 사용하여 순전파가 정상적으로 동작하는지 확인합니다."""
        custom_heights = [4, 3, 3, 3]
        custom_dilations = [1, 2, 3, 3]
        model = NestedUNet(
            in_ch=self.in_ch, out_ch=self.out_ch, width=16, heights=custom_heights, dilation_heights=custom_dilations
        )

        x = torch.randn(1, self.in_ch, 256, 256)
        try:
            output, *side_outputs = model(x)
            self.assertEqual(
                output.shape,
                (1, self.out_ch, 256, 256),
                "Final output shape mismatch with custom heights and dilations.",
            )
            for i, side in enumerate(side_outputs):
                self.assertEqual(
                    side.shape,
                    (1, self.out_ch, 256, 256),
                    f"Side output {i+1} shape mismatch with custom heights and dilations.",
                )
        except Exception as e:
            self.fail(f"Forward pass with custom heights and dilations failed with exception: {e}")

    def test_forward_pass_different_channels(self):
        """입력 및 출력 채널을 변경하여 모델이 올바르게 동작하는지 확인합니다."""
        in_ch = 1
        out_ch = 2
        model = NestedUNet(in_ch=in_ch, out_ch=out_ch, width=16, heights=[6, 5, 4, 3], dilation_heights=[1, 2, 3, 4])
        model.eval()
        x = torch.randn(1, in_ch, 256, 256)
        try:
            output, *side_outputs = model(x)
            self.assertEqual(
                output.shape, (1, out_ch, 256, 256), "Final output shape mismatch with different channels."
            )
            for i, side in enumerate(side_outputs):
                self.assertEqual(
                    side.shape, (1, out_ch, 256, 256), f"Side output {i+1} shape mismatch with different channels."
                )
        except Exception as e:
            self.fail(f"Forward pass with different channels failed with exception: {e}")

    def test_model_parameters_exist(self):
        """모델에 최소한 하나 이상의 파라미터가 존재하는지 확인합니다."""
        self.assertTrue(len(list(self.model.parameters())) > 0, "Model has no parameters.")

    def test_side_outputs_count(self):
        """생성되는 사이드 아웃풋의 개수가 레이어 수와 일치하는지 확인합니다."""
        x = torch.randn(1, self.in_ch, 256, 256)
        output, *side_outputs = self.model(x)
        self.assertEqual(
            len(side_outputs), self.model.num_layers, "Number of side outputs does not match number of layers."
        )

    def test_forward_pass_gpu(self):
        """GPU가 사용 가능한 경우, 모델이 GPU에서 정상적으로 동작하는지 확인합니다."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = NestedUNet(
                in_ch=self.in_ch,
                out_ch=self.out_ch,
                width=self.width,
                heights=self.heights,
                dilation_heights=self.dilation_heights,
            ).to(device)
            model.eval()
            x = torch.randn(1, self.in_ch, 256, 256).to(device)
            try:

                with torch.no_grad():
                    output, *side_outputs = model(x)
                self.assertEqual(output.shape, (1, self.out_ch, 256, 256), "Final output shape mismatch on GPU.")
                for i, side in enumerate(side_outputs):
                    self.assertEqual(
                        side.shape, (1, self.out_ch, 256, 256), f"Side output {i+1} shape mismatch on GPU."
                    )
            except Exception as e:
                self.fail(f"Forward pass on GPU failed with exception: {e}")
        else:
            self.skipTest("CUDA not available, skipping GPU forward pass test.")


if __name__ == "__main__":
    unittest.main()
