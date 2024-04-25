from importlib import import_module
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
from einops import parsing


class DimConverter:
    convertable = (
        nn.modules.conv._ConvNd,
        nn.modules.conv._ConvTransposeNd,
        nn.modules.pooling._LPPoolNd,
        nn.modules.pooling._MaxPoolNd,
        nn.modules.pooling._AvgPoolNd,
        nn.modules.pooling._AdaptiveAvgPoolNd,
        nn.modules.pooling._AdaptiveMaxPoolNd,
        nn.modules.pooling._MaxUnpoolNd,
        nn.modules.pooling._MaxUnpoolNd,
    )

    def __init__(self, model: nn.Module):
        # check input model's dimension
        self.current_dim = self.dim_analysis(model)
        # finding dimension-convertable layers
        self.layers = self.find_layers(model)

    @staticmethod
    def dim_analysis(model: nn.Module) -> int:
        # print(model)
        return 1

    def find_layers(self, model: nn.Module) -> dict[str, Union[nn.Module, torch.Tensor]]:
        """
        find dimension convertable layers in given model.
        return dictionary which has 'layer path' as key, and tuple of layer api and weight tensor as value.
        :param model:
        :return: Dict[str, tuple[nn.Module, torch.Tensor]]
        """
        weight_dict = dict(model.named_parameters())
        layer_dict = defaultdict(dict)

        for name, weight in weight_dict.items():
            if name.endswith("weight"):
                n = name.replace(".weight", "")
                w_name = "weight"
            elif name.endswith("bias"):
                n = name.replace(".bias", "")
                w_name = "bias"
            else:
                print(f"Unexpected name: {name}")
                n = ""
            layer = model.get_submodule(n)
            if isinstance(layer, self.convertable):
                layer_dict[n]["api"] = layer
                layer_dict[n][w_name] = weight

        return layer_dict

    def expand_dim(self):
        pass

    def reduce_dim(self):
        pass

    def __call__(self, model: nn.Module, pattern: str, *args, **kwargs):
        # 0. compare model's dimension with request pattern
        left, right = pattern.split("->")
        assert self.current_dim != left, f"Expect dimension {self.current_dim}, but receive {left}"
        l = parsing.ParsedExpression(left)
        r = parsing.ParsedExpression(right)

        # 1. convert dim of layers to appropriately
        if l == r:
            return model
        if l < r:
            re_layer = [self.expand_dim(l) for l in self.layers]
        else:
            re_layer = [self.reduce_dim(l) for l in self.layers]

        # 2. apply them to new model

        # 0. compare model's dimension with request pattern
