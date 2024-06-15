from importlib import import_module
from typing import OrderedDict, Union, overload

import torch.nn as nn
import torch

from ..builds.autoencoder import AutoEncoder
from ..converter import dim_converter
from ...nn_night.blocks import UNetBlock

checkpoint_map = {
    "brain": "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt",
    "mask": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    "scale0.5": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    "scale1.0": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth",
}

unet_map = {
    "brain": {"block": UNetBlock, "ch_in": 3, "ch_out": 1, "width": 32, "layers": [1, 1, 1, 1], "skip_connect": True},
    "mask": {"block": UNetBlock, "ch_in": 3, "ch_out": 2, "width": 64, "layers": [1, 1, 1, 1], "skip_connect": True},
}


def __brain_weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):
    for k, v in src_weight.items():
        if k.startswith(prefix := "encoder"):
            k = (
                k.replace(f"{prefix}1.enc1", f"{prefix}.body.0.0.")
                .replace(f"{prefix}2.enc2", f"{prefix}.body.1.0.")
                .replace(f"{prefix}3.enc3", f"{prefix}.body.2.0.")
                .replace(f"{prefix}4.enc4", f"{prefix}.body.3.0.")
            )
        elif k.startswith("bottleneck"):
            k = k.replace(f".bottleneck", ".")
        elif k.startswith("up"):
            k = (
                k.replace(f"upconv4", f"up_pools.0")
                .replace(f"upconv3", f"up_pools.1")
                .replace(f"upconv2", f"up_pools.2")
                .replace(f"upconv1", f"up_pools.3")
            )
        elif k.startswith(prefix := "decoder"):
            k = (
                k.replace(f"{prefix}1.dec1", f"{prefix}.3.0.")
                .replace(f"{prefix}2.dec2", f"{prefix}.2.0.")
                .replace(f"{prefix}3.dec3", f"{prefix}.1.0.")
                .replace(f"{prefix}4.dec4", f"{prefix}.0.0.")
            )
        else:
            k = k.replace("conv", "fc")

        if k in dst_weight.keys():
            dst_weight[k] = v
        else:
            print(k)

    return dst_weight


def unet(pretrained: bool = False):
    model = AutoEncoder(block=UNetBlock, ch_in=3, ch_out=1, width=32, layers=[1, 1, 1, 1], skip_connect=True)
    if pretrained:
        src_weight = torch.hub.load_state_dict_from_url(checkpoint_map["brain"], progress=True)
        dst_weight = __brain_weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model
