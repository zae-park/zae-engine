import torch.nn as nn
from einops import parsing


class DimConverter:
    def __init__(self, model: nn.Module):
        # check input model's dimension
        self.current_dim = self.dim_analysis(model)
        # finding dimension-convertable layers
        self.layers = self.find_layers(model)

    @staticmethod
    def dim_analysis(model: nn.Module) -> int:
        print(model)
        return 1

    @staticmethod
    def find_layers(model: nn.Module) -> list[nn.Module]:
        return []

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
