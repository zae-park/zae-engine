from typing import Callable, List, Type, Union, Tuple, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ...nn_night import blocks as blk


class CNNBase(nn.Module):
    def __init__(
        self,
        block: Type[Union[blk.BasicBlock, blk.Bottleneck, nn.Module]],
        ch_in: int,
        ch_out: int,
        width: int,
        layers: Sequence[int],
        groups: int = 1,
        dilation: int = 1,
        # zero_init_residual: bool = False,
        # replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.block = block
        self.ch_in = ch_in
        self.width = width
        self.ch_out = ch_out
        self.layers = layers
        self.norm_layer = norm_layer

        self.groups = groups
        self.dilation = dilation  # TODO: apply atrous convolution
        # self.base_width = width_per_group

        # make 'Stem' layer to receive input image and extract features with large kernel size as 7
        self.stem = self._make_stem(ch_in=ch_in, ch_out=width, kernel_size=7)

        # maks 'Body' layer with given 'block'. Expect that the 'block' include residual connection.
        body = []
        for i, l in enumerate(layers):
            ch_o = width * (2**i)
            ch_i = ch_o if i == 0 else ch_o * self.block.expansion // 2
            body.append(self.make_body(blocks=[block] * l, ch_in=ch_i, ch_out=ch_o, stride=2 if i else 1))
        self.body = nn.Sequential(*body)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 2 ** (len(self.layers) - 1) * block.expansion, ch_out)

        self.initializer()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     self.zero_initializer()

    def initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_initializer(self):
        for m in self.modules():
            if isinstance(m, blk.Bottleneck) and m.norm3.weight is not None:
                nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, blk.BasicBlock) and m.norm2.weight is not None:
                nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

    def get_output_shape(self, input_size: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate the encoder's output shape based on a dummy input.

        Parameters
        ----------
        input_size : Tuple[int, int, int, int]
            The size of the input tensor (batch_size, channels, height, width).

        Returns
        -------
        Tuple[int, int, int, int]
            The shape of the encoder's output tensor.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(input_size)
            features = self.forward(dummy_input)
            return features.shape

    def _make_stem(self, ch_in: int, ch_out: int, kernel_size: Union[int, tuple[int, int]]):
        conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=2, padding=3, bias=False)
        norm = self.norm_layer(ch_out)
        act = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return nn.Sequential(conv, norm, act, pool)

    def make_body(
        self,
        blocks: Union[List[Type[blk.BasicBlock | blk.Bottleneck]], tuple[Type[blk.BasicBlock | blk.Bottleneck]]],
        ch_in: int,
        ch_out: int,
        stride: int = 1,
    ) -> nn.Sequential:

        norm_layer = self.norm_layer

        layers = []
        # for 1st block
        block = blocks[0]
        layers.append(
            block(
                ch_in,
                ch_out,
                stride=stride,
                groups=self.groups,
                norm_layer=norm_layer,
            )
        )

        # from 2nd block to last
        for block in blocks[1:]:
            layers.append(
                block(
                    ch_out * self.block.expansion,
                    ch_out,
                    groups=self.groups,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        stem = self.stem(x)
        feat = self.body(stem)
        pool = torch.flatten(self.pool(feat), start_dim=1)
        out = self.fc(pool)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
