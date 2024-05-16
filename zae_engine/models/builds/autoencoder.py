from typing import Callable, Tuple, Type, Union
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
        kernel_size: Union[int, Tuple[int, int]],
        width: int,
        depth: int = 5,
        groups: int = 1,
        dilation: int = 1,
        # zero_init_residual: bool = False,
        # replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ):
        super(AutoEncoder, self).__init__()

        pools = nn.ModuleList()
        body = nn.ModuleList()

        for d in range(depth):
            if d == 0:
                body.append(
                    nn.Sequential(
                        self.unit_layer(ch_in, width, kernel_size),
                        self.gen_block(width, width, kernel_size, order),
                    )
                )
            else:
                if d == 1:
                    c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
                else:
                    c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
                pools.append(self._pool(p := (int(np.prod(stride[:d]))), p))
                body.append(
                    nn.Sequential(
                        self.unit_layer(c_in, c_out, kernel_size, stride=s),
                        self.gen_block(c_out, c_out, kernel_size, order),
                    )
                )

        self.encoder = []
        for i, l in enumerate(layers):
            ch_o = width * (2**i)
            ch_i = ch_o if i == 0 else ch_o * self.block.expansion // 2
            body.append(self._make_body(blocks=[block] * l, ch_in=ch_i, ch_out=ch_o, stride=2 if i else 1))
        self.body = nn.Sequential(*body)

        self.feature_vectors = []
        self.encoder.pool.register_forward_hook(self.feature_hook)

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

    def gen_block(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: Union[int, list, tuple],
        order: int,
        dilation: int = 1,
    ):
        """
        Return block, accumulated layers in-stage. Stack CBR 'order' times.
        :param ch_in: [Int] input channels of block
        :param ch_out: [Int] output channels of block
        :param kernel_size: [Int, Iterable] convolution kernel size.
            If type of argument is int, assume that model will receive 1-D input tensor.
            Else, the type of argument is iterable, assume that model will receive len(argument)-D input tensor.
        :param order: [Int] the number of blocks in stage.
            The stage means same resolution. e.g. from after previous pooling (or stem) to before next pooling.
        :param dilation: [Int, Iterable] Default is 1. If argument more than 1, dilated convolution will performed.
        :return: [nn.Module]
        """

        blk = []  # List of blocks
        for o in range(order):  # stack blocks 'order' times
            sequence = nnn.Residual(
                self.unit_layer(ch_in, ch_out, kernel_size, dilation=dilation),
                self.unit_layer(ch_out, ch_out, kernel_size, dilation=dilation),
            )
            blk.append(sequence)
        return nn.Sequential(*blk)

    def feature_hook(self, module, input_tensor, output_tensor):
        self.feature_vectors.append(input_tensor[0])

    # def feature_hook(self, module, input_tensor):
    #     self.feature_vectors.append(input_tensor[0])

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.bottleneck(self.feature_vectors.pop())

        dec4 = self.upconv4(feat)
        dec4 = torch.cat((dec4, self.feature_vectors.pop()), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.feature_vectors.pop()), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.feature_vectors.pop()), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.feature_vectors.pop()), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
