from copy import deepcopy
from typing import Union, Any, Dict
from collections import defaultdict

import torch
import torch.nn as nn
from einops import parsing
from torch.nn import Module


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

    def find_layers(self, model: nn.Module) -> defaultdict[str, dict[str, None | Module | torch.Tensor]]:
        """
        find dimension convertable layers in given model.
        return dictionary which has 'layer path' as key, and tuple of layer api and weight tensor as value.
        :param model:
        :return: Dict[str, tuple[nn.Module, torch.Tensor]]
        """
        layer_dict = defaultdict(lambda: {"api": None, "weight": None, "bias": None})

        for name, weight in model.named_parameters():
            name_split = name.split(".")
            if name_split[-1] not in ["weight", "bias"]:
                module_path, value_type = name, ""
            else:
                module_path, value_type = ".".join(name_split[:-1]), name_split[-1]

            module: Module = model.get_submodule(module_path)
            if isinstance(module, self.convertable):
                layer_dict[module_path]["api"] = module
                layer_dict[module_path][value_type] = weight

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
