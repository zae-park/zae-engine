from importlib import import_module
from typing import OrderedDict

import torch.nn as nn

from ..builds import cnn
from ..converter import dim_converter
from ...nn_night import blocks

res_map = {
    18: {"block": blocks.BasicBlock, "layers": [2, 2, 2, 2]},
    34: {"block": blocks.BasicBlock, "layers": [3, 4, 6, 3]},
    50: {"block": blocks.Bottleneck, "layers": [3, 4, 6, 3]},
    101: {"block": blocks.Bottleneck, "layers": [3, 4, 23, 3]},
    152: {"block": blocks.Bottleneck, "layers": [3, 8, 36, 3]},
}


def resnet_deco(n):
    """
    Decorator to wrap ResNet model creation functions with predefined configurations.

    Parameters
    ----------
    n : int
        The number of layers for the ResNet model.

    Returns
    -------
    function
        Wrapped function with ResNet configurations.
    """

    def wrapper(func):
        def wrap(*args, **kwargs):
            out = func(*args, **res_map[n], **kwargs)
            return out

        return wrap

    return wrapper


def __weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):
    """
    Map source weights to destination model weights.

    Parameters
    ----------
    src_weight : OrderedDict or dict
        Source model weights.
    dst_weight : OrderedDict or dict
        Destination model weights.

    Returns
    -------
    OrderedDict or dict
        Updated destination model weights.
    """

    for k, v in src_weight.items():

        if k.startswith("layer"):
            k = (
                k.replace("layer1", "body.0")
                .replace("layer2", "body.1")
                .replace("layer3", "body.2")
                .replace("layer4", "body.3")
            )
            k = k.replace(".bn", ".norm")
        elif k.startswith("fc"):
            pass
        else:
            k = "stem." + k
            k = k.replace("conv1", "0").replace("bn1", "1")

        dst_weight[k] = v

    return dst_weight


def resnet18(pretrained=False) -> cnn.CNNBase:
    """
    Create a ResNet-18 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, loads pre-trained weights from a specified checkpoint. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the ResNet-18 model.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """
    model = cnn.CNNBase(ch_in=3, ch_out=1000, width=64, groups=1, dilation=1, **res_map[18])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet18_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = __weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model


def resnet34(pretrained=False) -> cnn.CNNBase:
    """
    Create a ResNet-34 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, loads pre-trained weights from a specified checkpoint. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the ResNet-34 model.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """
    model = cnn.CNNBase(ch_in=3, ch_out=1000, width=64, groups=1, dilation=1, **res_map[34])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet34_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = __weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model


def resnet50(pretrained=False) -> cnn.CNNBase:
    """
    Create a ResNet-50 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, loads pre-trained weights from a specified checkpoint. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the ResNet-50 model.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """

    model = cnn.CNNBase(ch_in=3, ch_out=1000, width=64, groups=1, dilation=1, **res_map[50])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet50_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = __weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model


def resnet101(pretrained=False) -> cnn.CNNBase:
    """
    Create a ResNet-101 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, loads pre-trained weights from a specified checkpoint. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the ResNet-50 model.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """

    model = cnn.CNNBase(ch_in=3, ch_out=1000, width=64, groups=1, dilation=1, **res_map[101])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet101_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = __weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model


def resnet152(pretrained=False) -> cnn.CNNBase:
    """
    Create a ResNet-152 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, loads pre-trained weights from a specified checkpoint. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the ResNet-50 model.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    """

    model = cnn.CNNBase(ch_in=3, ch_out=1000, width=64, groups=1, dilation=1, **res_map[152])
    if pretrained:
        src_weight = import_module("torchvision.models").ResNet152_Weights.IMAGENET1K_V1.get_state_dict(True)
        dst_weight = __weight_mapper(src_weight, model.state_dict())
        model.load_state_dict(dst_weight, strict=True)
    return model


def se_injection(model: cnn.CNNBase) -> cnn.CNNBase:
    """
    Inject SE modules into the given ResNet model.

    Parameters
    ----------
    model : cnn.CNNBase
        The ResNet model to inject SE modules into.

    Returns
    -------
    cnn.CNNBase
        The ResNet model with SE modules injected.
    """
    for i, b in enumerate(model.body):
        for ii, blk in enumerate(b):
            if isinstance(blk, (blocks.BasicBlock, blocks.Bottleneck)):
                cvtr = dim_converter.DimConverter(blocks.SE1d(ch_in=blk.ch_out * blk.expansion))
                se_module = cvtr.convert("1d -> 2d")
                model.body[i][ii] = nn.Sequential(blk, se_module)
    return model


def cbam_injection(model: cnn.CNNBase) -> cnn.CNNBase:
    """
    Inject CBAM modules into the given ResNet model.

    Parameters
    ----------
    model : cnn.CNNBase
        The ResNet model to inject CBAM modules into.

    Returns
    -------
    cnn.CNNBase
        The ResNet model with SE modules injected.
    """
    for i, b in enumerate(model.body):
        for ii, blk in enumerate(b):
            if isinstance(blk, (blocks.BasicBlock, blocks.Bottleneck)):
                cvtr = dim_converter.DimConverter(blocks.CBAM1d(ch_in=blk.ch_out * blk.expansion))
                cbam_module = cvtr.convert("1d -> 2d")
                model.body[i][ii] = nn.Sequential(blk, cbam_module)
    return model


def seresnet18(pretrained=False) -> cnn.CNNBase:
    """
    Create an SE-ResNet-18 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for SE modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the SE-ResNet-18 model.

    References
    ----------
    Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-excitation networks.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    """
    model = resnet18(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for SE module.")
    model = se_injection(model)
    return model


def seresnet34(pretrained=False) -> cnn.CNNBase:
    """
    Create an SE-ResNet-34 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for SE modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the SE-ResNet-34 model.

    References
    ----------
    Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-excitation networks.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    """
    model = resnet34(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for SE module.")
    model = se_injection(model)
    return model


def seresnet50(pretrained=False) -> cnn.CNNBase:
    """
    Create an SE-ResNet-50 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for SE modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the SE-ResNet-50 model.

    References
    ----------
    Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-excitation networks.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    """

    model = resnet50(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for SE module.")
    model = se_injection(model)
    return model


def seresnet101(pretrained=False) -> cnn.CNNBase:
    """
    Create an SE-ResNet-101 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for SE modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the SE-ResNet-101 model.

    References
    ----------
    Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-excitation networks.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    """

    model = resnet101(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for SE module.")
    model = se_injection(model)
    return model


def seresnet152(pretrained=False) -> cnn.CNNBase:
    """
    Create an SE-ResNet-152 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for SE modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the SE-ResNet-152 model.

    References
    ----------
    Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-excitation networks.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    """
    model = resnet152(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for SE module.")
    model = se_injection(model)
    return model


def cbamresnet18(pretrained=False) -> cnn.CNNBase:
    """
    Create a CBAM-ResNet-18 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for CBAM modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the CBAM-ResNet-18 model.

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
           CBAM: Convolutional Block Attention Module.
           In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
           https://arxiv.org/abs/1807.06521
    """
    model = resnet18(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for CBAM module.")
    model = cbam_injection(model)
    return model


def cbamresnet34(pretrained=False) -> cnn.CNNBase:
    """
    Create a CBAM-ResNet-34 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for CBAM modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the CBAM-ResNet-34 model.

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
           CBAM: Convolutional Block Attention Module.
           In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
           https://arxiv.org/abs/1807.06521
    """
    model = resnet34(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for CBAM module.")
    model = cbam_injection(model)
    return model


def cbamresnet50(pretrained=False) -> cnn.CNNBase:
    """
    Create a CBAM-ResNet-50 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for CBAM modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the CBAM-ResNet-50 model.

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
           CBAM: Convolutional Block Attention Module.
           In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
           https://arxiv.org/abs/1807.06521
    """
    model = resnet50(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for CBAM module.")
    model = cbam_injection(model)
    return model


def cbamresnet101(pretrained=False) -> cnn.CNNBase:
    """
    Create a CBAM-ResNet-101 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for CBAM modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the CBAM-ResNet-101 model.

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
           CBAM: Convolutional Block Attention Module.
           In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
           https://arxiv.org/abs/1807.06521
    """
    model = resnet101(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for CBAM module.")
    model = cbam_injection(model)
    return model


def cbamresnet152(pretrained=False) -> cnn.CNNBase:
    """
    Create a CBAM-ResNet-152 model with the option to load pre-trained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, prints a message indicating no pre-trained weights for CBAM modules. Default is False.

    Returns
    -------
    cnn.CNNBase
        An instance of the CBAM-ResNet-152 model.

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
           CBAM: Convolutional Block Attention Module.
           In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
           https://arxiv.org/abs/1807.06521
    """
    model = resnet152(pretrained=pretrained)
    if pretrained:
        print("No pretrained weight for CBAM module.")
    model = cbam_injection(model)
    return model
