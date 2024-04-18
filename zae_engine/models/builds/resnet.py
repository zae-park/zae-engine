from functools import partial
from typing import Any, Callable, List, Type, Iterable, Union

import torch
import torch.nn as nn
from torch import Tensor

from blocks.resblock import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        ch_in: int,
        width: int,
        n_cls: int,
        layers: list[int],
        groups: int = 1,
        dilation: int = 1,
        # zero_init_residual: bool = False,
        # replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.ch_in = ch_in
        self.width = width
        self.n_cls = n_cls
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
            body.append(self._make_body(blocks=[block] * l, ch_in=width * (2**i), stride=2))
        self.body = nn.Sequential(*body)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * block.expansion, n_cls)

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
            if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_stem(self, ch_in: int, ch_out: int, kernel_size: Union[int, tuple[int, int]]):
        conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=2, padding=3, bias=False)
        norm = self.norm_layer(self.ch_in)
        act = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return nn.Sequential(conv, norm, act, pool)

    def _make_body(
        self,
        blocks: Iterable[BasicBlock | Bottleneck],
        ch_in: int,
        stride: int = 1,
    ) -> nn.Sequential:

        norm_layer = self.norm_layer
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(ch_in * 2, ch_in * 2, kernel_size=1, stride=stride),
                norm_layer(ch_in * 2),
            )

        layers = []
        # for 1st block
        block = blocks[0]
        layers.append(
            block(
                ch_in,
                ch_in * 2,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                norm_layer=norm_layer,
            )
        )

        # from 2nd block to last
        for block in blocks[1:]:
            layers.append(
                block(
                    ch_in,
                    ch_in * 2,
                    groups=self.groups,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == "__main__":
    resnet = ResNet(block=BasicBlock, ch_in=3, width=16, n_cls=10, layers=[2, 2, 2, 2])
    resnet = ResNet(block=Bottleneck, ch_in=6, width=16, n_cls=12, layers=[2, 3, 6, 2])
    resnet = ResNet(block=BasicBlock, ch_in=3, width=16, n_cls=10, layers=[2, 2, 2, 2], groups=2)
    resnet = ResNet(block=BasicBlock, ch_in=3, width=16, n_cls=10, layers=[2, 2, 2, 2], groups=2, dilation=2)
