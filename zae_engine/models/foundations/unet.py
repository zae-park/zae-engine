from typing import OrderedDict, Union

import torch

from ..builds import autoencoder
from ...nn_night import blocks

checkpoint_map = {
    "brain": "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt",
    "mask": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    "scale0.5": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    "scale1.0": "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth",
}

unet_map = {
    "brain": {
        "block": blocks.UNetBlock,
        "ch_in": 3,
        "ch_out": 1,
        "width": 32,
        "layers": [1, 1, 1, 1],
        "skip_connect": True,
    },
    "mask": {
        "block": blocks.UNetBlock,
        "ch_in": 3,
        "ch_out": 2,
        "width": 64,
        "layers": [1, 1, 1, 1],
        "skip_connect": True,
    },
}


def _brain_weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):
    """
    Map source weights to the destination model's weight dictionary, adjusting key names as needed.

    This function is used to map the keys of the pre-trained weights to the keys expected by the model.

    Parameters
    ----------
    src_weight : Union[OrderedDict, dict]
        Source model's state dictionary containing the pre-trained weights.
    dst_weight : Union[OrderedDict, dict]
        Destination model's state dictionary to which the pre-trained weights will be mapped.

    Returns
    -------
    Union[OrderedDict, dict]
        The updated destination model's state dictionary with the mapped pre-trained weights.
    """
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


def unet_brain(pretrained: bool = False) -> autoencoder.AutoEncoder:
    """
    Create a U-Net model with the option to load pre-trained weights.

    The U-Net model is a type of convolutional neural network developed for biomedical image segmentation.

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation,"
       in MICCAI 2015. (https://arxiv.org/abs/1505.04597)

    Parameters
    ----------
    pretrained : bool, optional
        If True, loads pre-trained weights from a specified checkpoint. Default is False.

    Returns
    -------
    zae_engine.models.autoencoder.AutoEncoder
        An instance of the AutoEncoder model with U-Net architecture.
    """
    model = autoencoder.AutoEncoder(
        block=blocks.UNetBlock, ch_in=3, ch_out=1, width=32, layers=[1, 1, 1, 1], skip_connect=True
    )
    if pretrained:
        src_weight = torch.hub.load_state_dict_from_url(checkpoint_map["brain"], progress=True)
        dst_weight = _brain_weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model
