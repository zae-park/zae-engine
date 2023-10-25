from typing import Optional

import torch.nn as nn

from .build import Segmentor1D, Regressor1D, ResNet1D
from .utility import load_weights, initializer, WeightLoader


def beat_segmentation(pretrained: Optional[bool] = False) -> nn.Module:
    """
    Build model which has a same structure with the latest released model.
    :param pretrained: bool
        If True, load weight from the server.
        If not, weights are initialized with 'initializer' method in utils.py
    :return: nn.Module
    """
    model = Segmentor1D(
        ch_in=1,
        ch_out=4,
        width=16,
        kernel_size=7,
        depth=5,
        order=4,
        stride=(2, 2, 2, 2),
        expanding=True,
        decoding=False,
    )

    if pretrained:
        weights = WeightLoader.get("beat_segmentation")
        model.load_state_dict(weights, strict=True)
    else:
        model.apply(initializer)

    return model


def peak_regression(pretrained: Optional[bool] = False) -> nn.Module:
    """
    Build model which has a same structure with the latest released model.
    :param pretrained: bool
        If True, load weight from the server.
        If not, weights are initialized with 'initializer' method in utils.py.
    :return: nn.Module
    """
    model = Regressor1D(
        dim_in=64,
        ch_in=1,
        width=16,
        kernel_size=3,
        depth=2,
        stride=1,
        order=4,
        head_depth=1,
        embedding_dims=16,
    )

    if pretrained:
        weights = WeightLoader.get("peak_regression")
        model.load_state_dict(weights, strict=True)
    else:
        model.apply(initializer)

    return model


def sec10_classification(pretrained: Optional[bool] = False) -> nn.Module:
    """
    Build model which has a same structure with the latest released model.
    :param pretrained: bool
        If True, load weight from the server.
        If not, weights are initialized randomly.
    :return: nn.Module
    """
    model = ResNet1D(
        num_layers=34,
        num_classes=7,
        num_channels=1,
        kernel_size=15,
        dropout_rate=0.3,
        width_factor=2,
        stride=2,
        reduction=4,
    )

    if pretrained:
        weights = WeightLoader.get("arrhythmia_classification")
        model.load_state_dict(weights, strict=True)

    return model
