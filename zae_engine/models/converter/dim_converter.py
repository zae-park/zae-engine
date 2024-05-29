import re
from copy import deepcopy
from typing import Union, Any, Dict
from collections import defaultdict

import torch
import torch.nn as nn
from einops import parsing
from torch.nn import Module


class DimConverter:
    __convertable = (
        nn.modules.conv._ConvNd,
        nn.modules.conv._ConvTransposeNd,
        nn.modules.pooling._LPPoolNd,
        nn.modules.pooling._MaxPoolNd,
        nn.modules.pooling._AvgPoolNd,
        nn.modules.pooling._AdaptiveAvgPoolNd,
        nn.modules.pooling._AdaptiveMaxPoolNd,
        nn.modules.pooling._MaxUnpoolNd,
        nn.modules.batchnorm._BatchNorm,
    )
    correction_map = {
        nn.Conv1d: nn.Conv2d,
        nn.Conv2d: nn.Conv3d,
        nn.ConvTranspose1d: nn.ConvTranspose2d,
        nn.ConvTranspose2d: nn.ConvTranspose3d,
        nn.LazyConvTranspose1d: nn.LazyConvTranspose2d,
        nn.LazyConvTranspose2d: nn.LazyConvTranspose3d,
        nn.MaxPool1d: nn.MaxPool2d,
        nn.MaxPool2d: nn.MaxPool3d,
        nn.AdaptiveMaxPool1d: nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool2d: nn.AdaptiveMaxPool3d,
        nn.AdaptiveAvgPool1d: nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool2d: nn.AdaptiveAvgPool3d,
        nn.BatchNorm1d: nn.BatchNorm2d,
        nn.BatchNorm2d: nn.BatchNorm3d,
    }  # default is expand mode

    def __init__(self, model: nn.Module):
        self.model = model
        # self.model.apply(self.dim_checker)
        # finding dimension-convertable layers
        self.module_dict = {}
        self.new_module_dict = {}
        self.layer_dict, self.param_dict = self.find_convertable(model)

    def dim_checker(self, module: nn.Module):
        """
        DEPRECATED
        """
        if isinstance(module, self.__convertable):
            module_dict = module.__dict__
            if "kernel_size" in module_dict.keys():
                if isinstance(module_dict["kernel_size"], int):
                    module.dim_check = 1
                else:
                    module.dim_check = len(module_dict["kernel_size"])
            elif "output_size" in module_dict.keys():
                if isinstance(module_dict["output_size"], int):
                    module.dim_check = 1
                else:
                    module.dim_check = len(module_dict["output_size"])

            else:
                print(f"Check unknown module {module}")
                module.dim_check = None
        else:
            module.dim_check = None

    def find_convertable(self, model: nn.Module) -> tuple[dict, dict]:
        """
        find dimension convertable layers in given model.
        return dictionary which has 'layer path' as key, and tuple of layer api and weight tensor as value.
        :param model:
        :return: tuple[dict, dict]
        """
        layer_dict = {}
        param_dict = {}

        self.module_dict = {}
        for name, weight in model.named_modules():
            module = model.get_submodule(name)
            if isinstance(module, self.__convertable):
                self.module_dict[name] = module

        for name, weight in model.named_parameters():
            param_dict[name] = weight
            if name.endswith(("weight", "bias")):
                root, leaf = name.rsplit(".", 1)
                module = model.get_submodule(root)
                if isinstance(module, self.__convertable):
                    layer_dict[root] = module

        return layer_dict, param_dict

    def dim_correction(self, reduce: bool):
        correction_map = {v: k for k, v in self.correction_map.items()} if reduce else self.correction_map
        correction_keys = tuple(correction_map.keys())
        for name, module in self.module_dict.items():
            if isinstance(module, correction_keys):
                corrected_layer = correction_map[type(module)]
                needs = corrected_layer.__init__.__annotations__
                ready = {k: v for k, v in self.const_getter(module, reduce=reduce).items() if k in needs.keys()}
                self.new_module_dict[name] = corrected_layer(**ready)

                # TODO: add weight correction
                # weight = module.weight
                # bias = module.bias

        # for k, v in self.layer_dict.items():
        #     corrected_layer = correction_map[type(v)]
        #     if isinstance(v, nn.MaxPool2d):
        #         print()
        #     needs = corrected_layer.__init__.__annotations__
        #     ready = {k: v for k, v in self.const_getter(v, reduce=reduce).items() if k in needs.keys()}
        #     self.layer_dict[k] = corrected_layer(**ready)
        #
        # for k, v in self.param_dict.items():
        #     if k.endswith("weight"):
        #         self.param_dict[k] = v.mean(-1) if reduce else torch.mm(v.unsqueeze(-1), v.unsqueeze(-2))

    @staticmethod
    def const_getter(conv_module: nn.Module, reduce: bool):
        module_dict = conv_module.__dict__
        const = {}
        for k in conv_module.__constants__:
            v = module_dict[k]
            if isinstance(v, tuple):
                v = v[:-1] if reduce else v[0:1] * (1 + len(v))
            const[k] = v
        return const

    def apply_new_dict(self, base_module: nn.Module, name: str, module: nn.Module):
        n = name.split(".")
        if len(n) == 1:
            base_module.add_module(name, module)
        else:
            self.apply_new_dict(base_module=base_module.get_submodule(n[0]), name=".".join(n[1:]), module=module)

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
        self.dim_correction(reduce=left > right)

        for k, v in self.new_module_dict.items():
            self.apply_new_dict(new_model, k, v)

        # TODO 2. apply them to new model
        # new_model.load_state_dict(self.param_dict)

        return new_model
