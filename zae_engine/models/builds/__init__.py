from .cnn import CNNBase
from .autoencoder import AutoEncoder, VAE
from .transformer import TransformerBase, BertBase, EncoderBase, DecoderBase
from .nested_autoencoder import NestedUNet

# from .legacy import ResNet1D, Regressor1D, Segmentor, CNNBaseLegacy

import torch


class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.body = torch.nn.Identity()
        self.param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        return self.body(x)
