# nn_night Sub-package

The `nn_night` sub-package of the **zae-engine** library is a collection of custom neural network layers and modules inspired by cutting-edge research in deep learning. This package offers a range of powerful and flexible components designed to enhance the functionality of deep learning models, with an emphasis on dynamic and adaptive mechanisms. This README provides an overview of the implemented components and how they can be used to build sophisticated neural network architectures.

## Implemented Components

### 1. ClippedReLU
The `ClippedReLU` activation function bounds the input tensor to a specified range, using upper and lower thresholds. This can be useful for controlling the output range of a layer, ensuring it remains within desired bounds.

### 2. GumbelSoftMax
The `GumbelSoftMax` layer implements the Gumbel-Softmax trick for categorical re-parameterization, allowing for differentiable sampling from a categorical distribution. This method is particularly useful in applications involving categorical latent variables.

### 3. Additional
The `Additional` module enables additional connections, allowing multiple input tensors to be passed through their respective sub-modules and summed together. It is a useful tool for designing network architectures that require residual or skip connections across multiple inputs.

### 4. DynOPool
`DynOPool` is a dynamic pooling layer that uses the Gumbel-Softmax trick to adaptively determine the pooling ratio. It allows the model to learn and adjust the pooling ratio during training, providing flexibility in feature extraction.

### 5. Inv1d
`Inv1d` is an involution layer for 1D inputs. It inverts the traditional convolution operation by using spatial aggregation and channel-wise interactions, inspired by recent advances in computer vision.

### 6. Positional Encoding Layers
The package includes several positional encoding layers for capturing positional information in sequence models:
- **SinusoidalPositionalEncoding**: Fixed positional encoding using sine and cosine functions.
- **LearnablePositionalEncoding**: Positional encoding learned during training.
- **RotaryPositionalEncoding**: Rotary encoding used for enhancing self-attention mechanisms.
- **RelativePositionalEncoding**: Captures relative distances between tokens, useful for language modeling.
- **AdaptivePositionalEncoding**: Adjusts positional encoding dynamically based on input sequence length.

### 7. Residual
The `Residual` module implements residual connections by adding the input tensor to the output of a sequence of layers, inspired by ResNet architectures.

### 8. SKConv1D
`SKConv1D` (Selective Kernel Convolution) is a convolutional layer for 1D inputs that uses multiple kernel sizes and a selection mechanism to adaptively fuse the features. It enables the model to focus on different receptive fields, improving feature extraction capabilities.

## Usage Examples
Below is an example of using the `ClippedReLU` layer and the `Residual` module in a simple neural network:

```python
import torch
import torch.nn as nn
from zae_engine.layers import ClippedReLU, Residual

# Define a simple model using ClippedReLU and Residual
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.activation = ClippedReLU(upper=1.0, lower=0.0)
        self.residual_block = Residual(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.residual_block(x)
        return x

# Example usage
model = SimpleModel()
x = torch.randn(5, 10)
output = model(x)
print(output)
```

This example shows how to use custom layers from `nn_night` to create a simple feedforward neural network with a residual block and a clipped ReLU activation function.

## Summary
The `nn_night` sub-package offers a range of powerful and flexible components for neural network design, inspired by recent advances in deep learning research. These modules can be used to create complex architectures that leverage dynamic, adaptive, and sophisticated mechanisms, providing the tools necessary to push the boundaries of modern deep learning models.