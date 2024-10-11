# Models Sub-package Overview

The Models sub-package of **zae-engine** library provides a wide range of pre-built neural network architectures and building blocks to facilitate deep learning research and development. This package contains multiple foundational implementations, such as AutoEncoder, CNN, Transformer, BERT, ResNet, and UNet, which can be used as-is or extended to create custom solutions.

## Key Implementations

### AutoEncoder and VAE

The **AutoEncoder** module provides a flexible implementation for both vanilla autoencoders and their variational counterparts (**VAE**). It features:

- **Flexible Encoder and Decoder Architecture**: The encoder and decoder can use various blocks, including ResNet and UNet blocks.
- **Skip Connections**: Optional skip connections allow for U-Net style encoding and decoding, beneficial in segmentation tasks.
- **Bottleneck Design**: The bottleneck layer between encoder and decoder can be customized with different layer types.

The VAE extends the basic AutoEncoder with a reparameterization trick to create latent variable distributions, allowing for generative capabilities.

### CNNBase

The **CNNBase** class implements a foundational convolutional neural network, providing flexibility in terms of:

- **Block Types**: Users can choose from basic blocks like **BasicBlock** and **Bottleneck**.
- **Custom Widths and Layers**: The network width and number of blocks per layer can be customized to control model depth and complexity.
- **Pooling and Fully Connected Layers**: The network ends with an adaptive pooling layer and a fully connected layer to generate the final output.

### TransformerBase and BertBase

The **TransformerBase** is a generic Transformer architecture that supports both encoder-only and encoder-decoder configurations.

- **Encoder-Only or Encoder-Decoder**: The model can work as an encoder-only model (e.g., BERT) or as an encoder-decoder (e.g., for translation tasks).
- **Flexible Components**: Supports custom embedding layers, encoders, and decoders, allowing users to implement transformers for various tasks.

The **BertBase** builds on TransformerBase and adds a pooling layer for the **[CLS]** token, making it similar to the original BERT architecture for tasks like classification.

### ResNet and UNet Foundations

- **ResNet**: Implementations for **ResNet** variants (18, 34, 50, 101, 152) are available. The **Bottleneck** and **BasicBlock** blocks are used as the building blocks for different versions. The models also provide the option to inject Squeeze-and-Excitation (**SE**) or **CBAM** modules for attention.
- **UNet**: The **UNet** foundation is implemented as a subclass of the AutoEncoder, featuring skip connections between encoder and decoder, particularly useful for biomedical image segmentation.

### Word Embedding Models

- **Word2VecEmbedding** and **FastTextEmbedding** classes allow users to leverage pre-trained **Word2Vec** and **FastText** embeddings from gensim in their deep learning models.

### Dimension Converter

The **DimConverter** class is provided to facilitate conversions of model layers from one dimension to another (e.g., 1D to 2D or 2D to 3D). This utility is particularly useful for adapting models to different data types, such as converting a text-processing model into an image-processing model.

## Example Usage

Below is an example of how to create a custom ResNet-50 model using the Models sub-package:

```python
from zae_engine.models.foundations import resnet50

# Create a ResNet-50 model instance
model = resnet50(pretrained=True)

# If required, inject SE or CBAM modules
from zae_engine.models.foundations.resnet import se_injection
model = se_injection(model)
```

To create a **VAE** model with U-Net style skip connections:

```python
from zae_engine.models.builds import autoencoder
from zae_engine.nn_night import blocks

vae_model = autoencoder.VAE(
    block=blocks.UNetBlock,
    ch_in=3,
    ch_out=1,
    width=64,
    layers=[2, 2, 2, 2],
    encoder_output_shape=[256, 16, 16],
    skip_connect=True,
    latent_dim=128
)
```

## Summary

The Models sub-package provides a variety of neural network architectures and utilities to support rapid prototyping and experimentation in deep learning. From foundational models like ResNet and BERT to specialized autoencoders, users have the flexibility to either use these models as-is or extend them according to their specific requirements.