from typing import Type, OrderedDict

from torchvision.models import (
    Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from ..builds.resnet import ResNet
from ..builds.blocks.resblock import BasicBlock, Bottleneck

res_map = {
    18: {"block": BasicBlock, "layers": [2, 2, 2, 2], "weight": ResNet18_Weights.IMAGENET1K_V1},
    34: {"block": BasicBlock, "layers": [3, 4, 6, 3], "weight": ResNet34_Weights.IMAGENET1K_V1},
    50: {"block": Bottleneck, "layers": [3, 4, 6, 3], "weight": ResNet50_Weights.IMAGENET1K_V1},
    101: {"block": Bottleneck, "layers": [3, 4, 23, 3], "weight": ResNet101_Weights.IMAGENET1K_V1},
    152: {"block": Bottleneck, "layers": [3, 8, 36, 3], "weight": ResNet152_Weights.IMAGENET1K_V1},
}


def resnet_deco(n):

    def wrapper(func):
        def wrap(*args, **kwargs):
            out = func(*args, **res_map[n], **kwargs)
            return out

        return wrap

    return wrapper


@resnet_deco(18)
def resnet18(pretrained=False, **kwargs):
    model = ResNet(block=kwargs["block"], ch_in=3, width=64, n_cls=1000, layers=kwargs["layers"], groups=1, dilation=1)
    if pretrained:
        model.load_state_dict(kwargs["weight"].get_state_dict(True))
    return model


@resnet_deco(34)
def resnet34(pretrained=False, **kwargs):
    model = ResNet(block=kwargs["block"], ch_in=3, width=64, n_cls=1000, layers=kwargs["layers"], groups=1, dilation=1)
    if pretrained:
        model.load_state_dict(kwargs["weight"].get_state_dict(True))
    return model


@resnet_deco(50)
def resnet50(pretrained=False, **kwargs):
    model = ResNet(block=kwargs["block"], ch_in=3, width=64, n_cls=1000, layers=kwargs["layers"], groups=1, dilation=1)
    if pretrained:
        model.load_state_dict(kwargs["weight"].get_state_dict(True))
    return model


@resnet_deco(101)
def resnet101(pretrained=False, **kwargs):
    model = ResNet(block=kwargs["block"], ch_in=3, width=64, n_cls=1000, layers=kwargs["layers"], groups=1, dilation=1)
    if pretrained:
        model.load_state_dict(kwargs["weight"].get_state_dict(True))
    return model


@resnet_deco(152)
def resnet152(pretrained=False, **kwargs):
    model = ResNet(block=kwargs["block"], ch_in=3, width=64, n_cls=1000, layers=kwargs["layers"], groups=1, dilation=1)
    if pretrained:
        model.load_state_dict(kwargs["weight"].get_state_dict(True))
    return model
