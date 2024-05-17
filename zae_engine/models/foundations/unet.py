from importlib import import_module
from typing import OrderedDict, Union

import torch.nn as nn
import torch

from ..builds.autoencoder import AutoEncoder
from ..converter import dim_converter
from ...nn_night.blocks import UNetBlock

checkpoint_map = {
    "scale0.5": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    "scale1.0": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth",
}


def weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):

    for k, v in src_weight.items():
        if k.startswith("encoder"):
            k = k.replace("encoder", "")
            body_count = int(k[0])
            k = (
                k.replace("layer1", "body.0")
                .replace("layer2", "body.1")
                .replace("layer3", "body.2")
                .replace("layer4", "body.3")
            )
        elif k.startswith("bottleneck"):
            pass
        elif k.startswith("decoder"):
            pass
        else:
            k = k.replace("conv", "fc")

        dst_weight[k] = v

    return dst_weight


def unet(pretrained: Union[str, bool] = False):
    model = AutoEncoder(block=UNetBlock, ch_in=3, ch_out=1, width=32, layers=[1, 1, 1, 1], groups=1, dilation=1)
    if pretrained:
        if isinstance(pretrained, str):
            if pretrained == "mask":
                torch_model = torch.hub.load_state_dict_from_url(checkpoint_map["scale0.5"], progress=True)
            elif pretrained == "brain":
                torch_model = torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet", pretrained=True)
            else:
                raise ValueError("Unexpected value")
        else:
            torch_model = torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet", pretrained=True)
        src_weight = torch_model.state_dict()
        dst_weight = weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight)
    return model
