# nn_night.blocks Sub-package Overview

The `nn_night.blocks` sub-package provides essential building blocks for constructing deep learning models, inspired by well-known architectures in computer vision and deep learning. These blocks are designed to facilitate model construction, leveraging proven techniques such as residual connections, attention mechanisms, and modularity for flexible and effective deep learning applications.

### Implemented Components

#### 1. BasicBlock

The `BasicBlock` class represents a standard residual block, as introduced by He et al. in their ResNet paper [1]_. It includes two convolutional layers with ReLU activations and a residual connection that adds the input to the output. If the dimensions do not match, a downsampling layer is applied to align them.

- **Parameters**:
  - `ch_in`: Number of input channels.
  - `ch_out`: Number of output channels.
  - `stride`, `groups`, `dilation`: Options for convolution operations.
  - `norm_layer`: Normalization layer, defaulting to `nn.BatchNorm2d`.

#### 2. Bottleneck

The `Bottleneck` class is a deeper variant of the `BasicBlock` that uses three convolutional layers to create a more efficient residual unit, particularly for architectures like ResNet-50 and beyond. This block is also based on the ResNet paper [1]_.

- **Parameters**:
  - Similar to `BasicBlock`, but includes an expansion parameter (`expansion=4`) to increase the number of output channels.

#### 3. SE1d (Squeeze and Excitation Block)

The `SE1d` class implements the Squeeze and Excitation (SE) block for 1D inputs, as proposed by Hu et al. [2]_. This block adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.

- **Parameters**:
  - `ch_in`: Number of input channels.
  - `reduction`: Reduction ratio for the SE block (default is 8).
  - `bias`: Whether to use bias in the fully connected layers.

#### 4. CBAM1d (Convolutional Block Attention Module)

The `CBAM1d` class implements the Convolutional Block Attention Module for 1D inputs, providing both channel-wise and spatial attention mechanisms as introduced by Woo et al. [3]_. This module is useful for enhancing feature representations by focusing on important regions of the input.

- **Parameters**:
  - `ch_in`: Number of input channels.
  - `kernel_size`, `reduction`, `bias`, `conv_pool`: Options for controlling convolution, pooling, and reduction.

#### 5. UNetBlock

The `UNetBlock` class is a modification of the `BasicBlock` specifically for use in UNet-like architectures. The block combines residual connections with UNet downsampling strategies, making it well-suited for biomedical image segmentation tasks as described by Ronneberger et al. [4]_.

- **Parameters**:
  - `ch_in`, `ch_out`: Input and output channels.
  - `stride`, `groups`, `norm_layer`: Configuration options similar to `BasicBlock`.

#### 6. Spatial Attention Mechanisms

- **SE1d** and **CBAM1d** both implement spatial attention mechanisms that help improve model performance by allowing the network to focus on the most relevant parts of the input. SE1d focuses on channel recalibration, while CBAM1d combines channel and spatial attention.

### How to Use the Blocks

The blocks in `nn_night.blocks` can be easily integrated into any PyTorch model to add residual learning, attention, or UNet-style processing capabilities. Below is an example of how to use a `BasicBlock` in a custom neural network model:

```python
import torch
import torch.nn as nn
from nn_night.blocks.resblock import BasicBlock

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = BasicBlock(ch_in=64, ch_out=128, stride=2)
        self.layer2 = BasicBlock(ch_in=128, ch_out=128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = CustomModel()
print(model)
```

This example demonstrates how to construct a simple neural network using the `BasicBlock`. The same principles can be applied to the other blocks like `Bottleneck`, `SE1d`, or `UNetBlock`, depending on the requirements of your model.

### Summary

The `nn_night.blocks` sub-package is a set of versatile components inspired by popular deep learning architectures. These components provide an easy way to incorporate advanced mechanisms like residual learning, bottleneck structures, and attention modules, facilitating the construction of state-of-the-art models.

### References

.. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR) (pp. 770-778). https://arxiv.org/abs/1512.03385

.. [2] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR) (pp. 7132-7141). https://arxiv.org/abs/1709.01507

.. [3] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19). https://arxiv.org/abs/1807.06521

.. [4] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). https://arxiv.org/abs/1505.04597

