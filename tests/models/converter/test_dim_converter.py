import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from zae_engine.models.converter import DimConverter
from zae_engine.models.foundations import resnet18


class TestDimConverter(unittest.TestCase):
    def setUp(self):
        self.model_2d = resnet18()
        self.converter_2d_to_1d = DimConverter(self.model_2d)

    def test_find_convertable(self):
        layer_dict, param_dict = self.converter_2d_to_1d.find_convertable(self.model_2d)
        self.assertTrue("body.0.0.conv1" in layer_dict)
        self.assertTrue("body.0.1.conv2" in layer_dict)
        self.assertTrue("body.1.0.downsample.0" in layer_dict)

        self.assertTrue("body.0.0.conv1.weight" in param_dict)
        self.assertTrue("body.0.1.conv2.bias" in param_dict)
        self.assertTrue("body.1.0.downsample.0.weight" in param_dict)

    def test_dim_correction(self):
        self.converter_2d_to_1d.dim_correction(reduce=True)
        for name, module in self.converter_2d_to_1d.new_module_dict.items():
            self.assertTrue(isinstance(module, (nn.Conv1d, nn.BatchNorm1d, nn.AdaptiveAvgPool1d, nn.MaxPool1d)))

    def test_convert_and_run(self):
        new_model = self.converter_2d_to_1d.convert("2d -> 1d")
        self.assertTrue(isinstance(new_model.body[0][0].conv1, nn.Conv1d))
        self.assertTrue(isinstance(new_model.body[1][0].downsample[0], nn.Conv1d))
        self.assertTrue(isinstance(new_model.body[1][0].downsample[1], nn.BatchNorm1d))

        # Create dummy 1D input data
        input_data = torch.randn(1, 3, 128)  # 1D input with shape (batch_size, channels, length)
        target = torch.randint(0, 10, (1,))  # Dummy target for backward pass

        # Run forward pass
        output = new_model(input_data)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        # Run backward pass
        loss.backward()

        # Check if gradients are computed
        for param in new_model.parameters():
            self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
