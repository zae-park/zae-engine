import re
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
        self.model = model
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

    def expand_dim(
        self, src_dict: dict[str, Union[nn.Module, torch.Tensor]]
    ) -> dict[str, Union[nn.Module, torch.Tensor]]:
        expand_map = {nn.Conv2d: nn.Conv3d, nn.Conv1d: nn.Conv2d}
        dst_dict = {}
        for k, v in src_dict.items():
            for kk, vv in v.items():
                if kk == "weight":
                    dst_dict[f"{k}.{kk}"] = torch.mm(vv.unsqueeze(-1), vv.unsqueeze(-2))
                elif kk == "bias":
                    dst_dict[f"{k}.{kk}"] = vv
                else:
                    dst_api = expand_map[type(vv)]
                    needs = dst_api.__init__.__annotations__
                    ready = {k: v for k, v in self.const_getter(vv, reduce=False).items() if k in needs.keys()}
                    dst_dict[k] = dst_api(**ready)
        return dst_dict

    def reduce_dim(
        self, src_dict: dict[str, Union[nn.Module, torch.Tensor]]
    ) -> dict[str, Union[nn.Module, torch.Tensor]]:
        reduce_map = {nn.Conv3d: nn.Conv2d, nn.Conv2d: nn.Conv1d}

        dst_dict = {}
        for k, v in src_dict.items():
            for kk, vv in v.items():
                if kk == "weight":
                    dst_dict[f"{k}.{kk}"] = vv.mean(-1)
                elif kk == "bias":
                    dst_dict[f"{k}.{kk}"] = vv
                else:
                    dst_api = reduce_map[type(vv)]
                    needs = dst_api.__init__.__annotations__
                    ready = {k: v for k, v in self.const_getter(vv, reduce=True).items() if k in needs.keys()}
                    dst_dict[k] = dst_api(**ready)
        return dst_dict

    @staticmethod
    def const_getter(conv_module: nn.Module, reduce: bool):
        module_dict = conv_module.__dict__
        # const = {k: module_dict[k] for k in conv_module.__constants__}
        const = {}
        for k in conv_module.__constants__:
            v = module_dict[k]
            if isinstance(v, tuple):
                v = v[:-1] if reduce else tuple(list(v) + [v[:-1]])
            const[k] = v
        return const

    def convert(self, pattern: str, *args, **kwargs):
        # 0. compare model's dimension with request pattern
        left, right = pattern.split("->")
        # assert self.current_dim != left, f"Expect dimension {self.current_dim}, but receive {left}"
        left = left.strip()
        right = right.strip()

        new_model = deepcopy(self.model)

        if left == right:
            return new_model

        # convert dim of layers to appropriately
        new_dict = self.reduce_dim(self.layers) if left > right else self.expand_dim(self.layers)

        # 2. apply them to new model
        new_model.load_state_dict(new_dict)
        return new_model
