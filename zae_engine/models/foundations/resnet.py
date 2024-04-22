from importlib import import_module
from typing import OrderedDict

from ..builds.cnn import CNNBase
from zae_engine.nn_night.blocks import BasicBlock, Bottleneck

res_map = {
    18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
    34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
    50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
    101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
    152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
}


def resnet_deco(n):

    def wrapper(func):
        def wrap(*args, **kwargs):
            out = func(*args, **res_map[n], **kwargs)
            return out

        return wrap

    return wrapper


def weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):

    for k, v in src_weight.items():

        if k.startswith("layer"):
            k = (
                k.replace("layer1", "body.0")
                .replace("layer2", "body.1")
                .replace("layer3", "body.2")
                .replace("layer4", "body.3")
            )
        elif k.startswith("fc"):
            pass
        else:
            k = "stem." + k
            k = k.replace("conv1", "0").replace("bn1", "1")

        dst_weight[k] = v

    return dst_weight


def resnet18(pretrained=False):
    model = CNNBase(ch_in=3, width=64, n_cls=1000, groups=1, dilation=1, **res_map[18])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet18_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight)
    return model


def resnet34(pretrained=False):
    model = CNNBase(ch_in=3, width=64, n_cls=1000, groups=1, dilation=1, **res_map[34])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet34_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight)
    return model


def resnet50(pretrained=False):
    model = CNNBase(ch_in=3, width=64, n_cls=1000, groups=1, dilation=1, **res_map[50])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet50_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight)
    return model


def resnet101(pretrained=False):
    model = CNNBase(ch_in=3, width=64, n_cls=1000, groups=1, dilation=1, **res_map[101])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet101_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight)
    return model


def resnet152(pretrained=False):
    model = CNNBase(ch_in=3, width=64, n_cls=1000, groups=1, dilation=1, **res_map[152])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet152_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight)
    return model
