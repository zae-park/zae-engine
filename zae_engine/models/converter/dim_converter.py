import re
from copy import deepcopy
from typing import Union, Any, Dict
from collections import defaultdict

import torch
import torch.nn as nn
from einops import parsing
from torch.nn import Module


class DimConverter:
    """
    A class to convert the dimensionality of layers in a given PyTorch model from 1D to 2D, 2D to 3D, or vice versa.

    This class identifies layers in a model that can be converted to different dimensions, such as Conv1d to Conv2d,
    and adapts their weights and parameters accordingly.

    Attributes
    ----------
    model : nn.Module
        The original PyTorch model to be converted.
    module_dict : dict
        A dictionary storing the original layers of the model.
    new_module_dict : dict
        A dictionary storing the converted layers of the model.
    layer_dict : dict
        A dictionary mapping layer names to their respective modules.
    param_dict : dict
        A dictionary mapping parameter names to their respective tensors.

    Methods
    -------
    find_convertable(model: nn.Module) -> tuple[dict, dict]:
        Finds and returns layers and parameters in the model that can be converted.
    dim_correction(reduce: bool):
        Corrects the dimensionality of the layers identified for conversion.
    const_getter(conv_module: nn.Module, reduce: bool) -> dict:
        Retrieves and adjusts the constants of a given convolutional module.
    apply_new_dict(base_module: nn.Module, name: str, module: nn.Module):
        Applies the converted layers to a new model structure.
    convert(pattern: str, *args, **kwargs) -> nn.Module:
        Converts the dimensionality of the model based on the specified pattern.
    """

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
        """
        Initialize the DimConverter with the given model.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to be converted.
        """
        self.model = model
        self.module_dict = {}
        self.new_module_dict = {}
        self.layer_dict, self.param_dict = self.find_convertable(model)

    def find_convertable(self, model: nn.Module) -> tuple[dict, dict]:
        """
        Find dimension-convertable layers in the given model.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to be analyzed.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing two dictionaries: layer_dict and param_dict.
            layer_dict maps layer names to modules.
            param_dict maps parameter names to tensors.
        """
        layer_dict = {}
        param_dict = {}

        self.module_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, self.__convertable):
                self.module_dict[name] = module

        for name, param in model.named_parameters():
            param_dict[name] = param
            if name.endswith(("weight", "bias")):
                root, _ = name.rsplit(".", 1)
                module = model.get_submodule(root)
                if isinstance(module, self.__convertable):
                    layer_dict[root] = module

        return layer_dict, param_dict

    def dim_correction(self, reduce: bool):
        """
        Correct the dimensionality of the layers identified for conversion.

        Parameters
        ----------
        reduce : bool
            Whether to reduce the dimensionality (e.g., 2D to 1D) or expand it (e.g., 1D to 2D).
        """
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

    @staticmethod
    def const_getter(conv_module: nn.Module, reduce: bool) -> dict:
        """
        Retrieve and adjust the constants of a given convolutional module.

        Parameters
        ----------
        conv_module : nn.Module
            The convolutional module to be adjusted.
        reduce : bool
            Whether to reduce the dimensionality (e.g., 2D to 1D) or expand it (e.g., 1D to 2D).

        Returns
        -------
        dict
            A dictionary of adjusted constants.
        """
        module_dict = conv_module.__dict__
        const = {}
        for k in conv_module.__constants__:
            v = module_dict[k]
            if isinstance(v, tuple):
                v = v[:-1] if reduce else v[0:1] * (1 + len(v))
            const[k] = v
        return const

    def apply_new_dict(self, base_module: nn.Module, name: str, module: nn.Module):
        """
        Apply the converted layers to a new model structure.

        Parameters
        ----------
        base_module : nn.Module
            The base module to which the converted layers will be applied.
        name : str
            The name of the layer to be applied.
        module : nn.Module
            The converted module to be applied.
        """
        n = name.split(".")
        if len(n) == 1:
            base_module.add_module(name, module)
        else:
            self.apply_new_dict(base_module=base_module.get_submodule(n[0]), name=".".join(n[1:]), module=module)

    def convert(self, pattern: str, *args, **kwargs) -> nn.Module:
        """
        Convert the dimensionality of the model based on the specified pattern.

        Parameters
        ----------
        pattern : str
            The conversion pattern (e.g., "1d -> 2d", "2d -> 3d").

        Returns
        -------
        nn.Module
            The converted PyTorch model.
        """
        left, right = pattern.split("->")
        left = left.strip()
        right = right.strip()

        new_model = deepcopy(self.model)

        if left == right:
            return new_model

        self.dim_correction(reduce=left > right)

        for k, v in self.new_module_dict.items():
            self.apply_new_dict(new_model, k, v)

        # TODO 2. apply them to new model
        # new_model.load_state_dict(self.param_dict)

        return new_model
    
    # def convert_weights(self, original_module: nn.Module, corrected_module: nn.Module, reduce: bool):
    #     """
    #     Converts and assigns weights from the original module to the corrected module.

    #     Parameters
    #     ----------
    #     original_module : nn.Module
    #         The original convolutional module.
    #     corrected_module : nn.Module
    #         The corrected convolutional module with different dimensionality.
    #     reduce : bool
    #         Whether the dimensionality is being reduced or expanded.
    #     """
    #     with torch.no_grad():
    #         if isinstance(original_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
    #             # Handle Conv layers
    #             if reduce:
    #                 # e.g., Conv2d -> Conv1d: average over the last dimension (e.g., kernel_w)
    #                 # Original weight shape: (out_channels, in_channels, kH, kW)
    #                 # New weight shape: (out_channels, in_channels, k)
    #                 # Assuming kernel_size is being reduced from (kH, kW) to k
    #                 # Here, we'll average over the last dimension (kW)
    #                 if isinstance(original_module, nn.Conv1d):
    #                     # Conv1d cannot be reduced further, skip
    #                     return
    #                 elif isinstance(original_module, nn.Conv2d):
    #                     # Conv2d -> Conv1d
    #                     # Take the mean over the last spatial dimension (kW)
    #                     original_weight = original_module.weight.data
    #                     new_weight = original_weight.mean(dim=3)  # Shape: (out_channels, in_channels, kH)
    #                     # If kH > 1, you might want to further reduce or handle accordingly
    #                     # For simplicity, let's assume kH=1 after reduction
    #                     if new_weight.shape[2] != 1:
    #                         # Average over kH as well to get a single dimension
    #                         new_weight = new_weight.mean(dim=2)  # Shape: (out_channels, in_channels, 1)
    #                     corrected_module.weight.data = new_weight
    #                     # If bias exists, copy it
    #                     if original_module.bias is not None and corrected_module.bias is not None:
    #                         corrected_module.bias.data = original_module.bias.data
    #             else:
    #                 # e.g., Conv1d -> Conv2d: replicate weights along the new dimension
    #                 # Original weight shape: (out_channels, in_channels, k)
    #                 # New weight shape: (out_channels, in_channels, kH, kW)
    #                 if isinstance(original_module, nn.Conv1d):
    #                     # Conv1d -> Conv2d
    #                     original_weight = original_module.weight.data
    #                     # Decide on kernel_size for new dimension, e.g., (k, 1) or (1, k)
    #                     # Here, we'll expand to (k, k) by replicating
    #                     k = original_module.kernel_size[0]
    #                     new_kH, new_kW = k, k  # Example: (k, k)
    #                     # Expand the weight
    #                     new_weight = original_weight.unsqueeze(-1).repeat(1, 1, 1, new_kW)  # Shape: (out_channels, in_channels, k, k)
    #                     corrected_module.weight.data = new_weight
    #                     # If bias exists, copy it
    #                     if original_module.bias is not None and corrected_module.bias is not None:
    #                         corrected_module.bias.data = original_module.bias.data
    #                 elif isinstance(original_module, nn.Conv2d):
    #                     # Conv2d -> Conv3d
    #                     original_weight = original_module.weight.data
    #                     # Decide on kernel_size for new dimension, e.g., (k, 1, 1)
    #                     # Here, we'll expand to (k, kH, kW) by replicating
    #                     kH, kW = original_module.kernel_size
    #                     new_kD = 1  # Example: add a new dimension with size 1
    #                     new_weight = original_weight.unsqueeze(2).repeat(1, 1, new_kD, 1, 1)  # Shape: (out_channels, in_channels, kD, kH, kW)
    #                     # However, Conv3d expects (out_channels, in_channels, kD, kH, kW)
    #                     if isinstance(corrected_module, nn.Conv3d):
    #                         corrected_module.weight.data = new_weight.squeeze(2)  # Adjust as needed
    #                     # If bias exists, copy it
    #                     if original_module.bias is not None and corrected_module.bias is not None:
    #                         corrected_module.bias.data = original_module.bias.data

    #         elif isinstance(original_module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
    #             # Handle ConvTranspose layers similarly
    #             if reduce:
    #                 if isinstance(original_module, nn.ConvTranspose2d):
    #                     # ConvTranspose2d -> ConvTranspose1d
    #                     original_weight = original_module.weight.data
    #                     new_weight = original_weight.mean(dim=3)  # Shape: (out_channels, in_channels, kH)
    #                     if new_weight.shape[2] != 1:
    #                         new_weight = new_weight.mean(dim=2)  # Shape: (out_channels, in_channels, 1)
    #                     corrected_module.weight.data = new_weight
    #                     if original_module.bias is not None and corrected_module.bias is not None:
    #                         corrected_module.bias.data = original_module.bias.data
    #             else:
    #                 if isinstance(original_module, nn.ConvTranspose1d):
    #                     # ConvTranspose1d -> ConvTranspose2d
    #                     original_weight = original_module.weight.data
    #                     k = original_module.kernel_size[0]
    #                     new_kH, new_kW = k, k
    #                     new_weight = original_weight.unsqueeze(-1).repeat(1, 1, 1, new_kW)  # Shape: (out_channels, in_channels, kH, kW)
    #                     corrected_module.weight.data = new_weight
    #                     if original_module.bias is not None and corrected_module.bias is not None:
    #                         corrected_module.bias.data = original_module.bias.data

    #         elif isinstance(original_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #             # Handle BatchNorm layers by copying the parameters directly
    #             corrected_module.weight.data = original_module.weight.data.clone()
    #             corrected_module.bias.data = original_module.bias.data.clone()
    #             corrected_module.running_mean = original_module.running_mean.clone()
    #             corrected_module.running_var = original_module.running_var.clone()
    #             if hasattr(corrected_module, 'num_batches_tracked') and hasattr(original_module, 'num_batches_tracked'):
    #                 corrected_module.num_batches_tracked = original_module.num_batches_tracked.clone()