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
        skip_connect: bool = False,
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
        self.skip_connect = skip_connect

        self.encoder.stem = nn.Identity()
        self.encoder.body[0] = self.encoder.make_body(blocks=[block] * layers[0], ch_in=ch_in, ch_out=width, stride=2)

        self.feature_vectors = []
        # [U-net] Register hook for every blocks in encoder when "skip_connect" is true.
        if skip_connect:
            for b in self.encoder.body:
                b[0].relu2.register_forward_hook(self.feature_hook)
        # registrate hook to end of last body instead of pooling layer
        self.encoder.body[-1].register_forward_hook(self.feature_output_hook)

        self.bottleneck = block(width * 8, width * 16)

        up_pools = []
        decoder = []
        for i, l in enumerate(layers):
            c_i, c_o = width * 2 ** (i + 1), width * 2**i
            up_pools.append(nn.ConvTranspose2d(in_channels=c_i, out_channels=c_o, kernel_size=2, stride=2))
            decoder.append(self.encoder.make_body([block] * l, ch_in=c_i if skip_connect else c_i // 2, ch_out=c_o))
        self.up_pools = nn.ModuleList(reversed(up_pools))
        self.decoder = nn.ModuleList(reversed(decoder))

        self.fc = nn.Conv2d(in_channels=width, out_channels=ch_out, kernel_size=1)
        self.sig = nn.Sigmoid()

    def feature_hook(self, module, input_tensor, output_tensor):
        self.feature_vectors.append(input_tensor[0])

    def feature_output_hook(self, module, input_tensor, output_tensor):
        self.feature_vectors.append(output_tensor)

    def forward(self, x):
        feat = self.encoder(x)  # Forwarding encoder & hook immediate outputs
        feat = self.bottleneck(self.feature_vectors.pop())

        for up_pool, dec in zip(self.up_pools, self.decoder):
            feat = up_pool(feat)
            if self.skip_connect:
                feat = torch.cat((feat, self.feature_vectors.pop()), dim=1)
            feat = dec(feat)
        return self.sig(self.fc(feat))
