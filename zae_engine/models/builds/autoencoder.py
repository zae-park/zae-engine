from typing import Callable, Tuple, Type, Union, List
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from . import cnn
from ...nn_night import blocks as blk


class AutoEncoder(nn.Module):

    def __init__(
        self,
        block: Type[Union[blk.UNetBlock, nn.Module]],
        ch_in: int,
        ch_out: int,
        width: int,
        layers: Union[Tuple[int], List[int]],
        groups: int = 1,
        dilation: int = 1,
        # zero_init_residual: bool = False,
        # replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        skip_connection: bool = False,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder = cnn.CNNBase(
            block=block,
            ch_in=ch_in,
            ch_out=ch_out,
            width=width,
            layers=layers,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        # Remove Stem layer in CNNBase & use 1st layer in body instead.
        self.skip_connection = skip_connection

        self.encoder.stem = nn.Identity()
        self.encoder.body[0] = self.encoder.make_body(blocks=[block] * layers[0], ch_in=ch_in, ch_out=width, stride=2)

        self.feature_vectors = []
        self.encoder.pool.register_forward_hook(self.feature_hook)
        # [U-net] Register hook for every blocks in encoder when "skip_connection" is true.
        if skip_connection:
            for b in self.encoder.body:
                b[0].relu2.register_forward_hook(self.feature_hook)

        self.bottleneck = block(width * 8, width * 16)

        self.upconv4 = nn.ConvTranspose2d(width * 16, width * 8, kernel_size=2, stride=2)
        self.decoder4 = block((width * 8) * 2, width * 8)
        self.upconv3 = nn.ConvTranspose2d(width * 8, width * 4, kernel_size=2, stride=2)
        self.decoder3 = block((width * 4) * 2, width * 4)
        self.upconv2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.decoder2 = block((width * 2) * 2, width * 2)
        self.upconv1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.decoder1 = block(width * 2, width)

        self.conv = nn.Conv2d(in_channels=width, out_channels=ch_out, kernel_size=1)

    def feature_hook(self, module, input_tensor, output_tensor):
        self.feature_vectors.append(input_tensor[0])

    # def feature_hook(self, module, input_tensor):
    #     self.feature_vectors.append(input_tensor[0])

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.bottleneck(self.feature_vectors.pop())

        dec4 = self.upconv4(feat)
        if self.skip_connection:
            dec4 = torch.cat((dec4, self.feature_vectors.pop()), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        if self.skip_connection:
            dec3 = torch.cat((dec3, self.feature_vectors.pop()), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        if self.skip_connection:
            dec2 = torch.cat((dec2, self.feature_vectors.pop()), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        if self.skip_connection:
            dec1 = torch.cat((dec1, self.feature_vectors.pop()), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
