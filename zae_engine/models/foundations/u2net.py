from typing import OrderedDict, Union, Sequence, Callable

import torch
import torch.nn as nn

import math

# Import necessary modules from your project structure
from zae_engine.models import AutoEncoder
from zae_engine.nn_night import blocks as blk

__all__ = ["U2NET_full", "U2NET_lite"]


def _upsample_like(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Upsample a tensor to a given size using bilinear interpolation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be upsampled.
    size : tuple
        Desired output size (height, width).

    Returns
    -------
    torch.Tensor
        Upsampled tensor.
    """
    return nn.Upsample(size=size, mode="bilinear", align_corners=False)(x)


def _size_map(x: torch.Tensor, height: int) -> dict:
    """
    Generate a mapping from each depth level to the spatial size for upsampling.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    height : int
        Number of levels in the U-structure.

    Returns
    -------
    dict
        A dictionary mapping each level to its corresponding spatial size.
    """
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    """
    Residual Block with Convolution, Batch Normalization, and ReLU Activation.

    This module performs a convolution followed by batch normalization and ReLU activation.
    It serves as a fundamental building block in the RSU (Residual U-block) structure.

    Parameters
    ----------
    in_ch : int, optional
        Number of input channels. Default is 3.
    out_ch : int, optional
        Number of output channels. Default is 3.
    dilate : int, optional
        Dilation rate for the convolution. Default is 1.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, dilate: int = 1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the REBNCONV block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_ch, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_ch, height, width).
        """
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    """
    Residual U-block (RSU) used in the U^2-Net architecture.

    The RSU block is a nested U-structure that allows the network to capture multi-scale features.
    It consists of multiple layers of convolutional blocks arranged in a U-shaped pattern.

    Parameters
    ----------
    name : str
        Name identifier for the RSU block.
    height : int
        Number of layers in the RSU block.
    in_ch : int
        Number of input channels.
    mid_ch : int
        Number of channels in the middle layers.
    out_ch : int
        Number of output channels.
    dilated : bool, optional
        Whether to use dilated convolutions. Default is False.
    """

    def __init__(self, name: str, height: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool = False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RSU block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_ch, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the RSU block.
        """
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        def unet(x_inner: torch.Tensor, height_inner: int = 1) -> torch.Tensor:
            """
            Recursive U-Net-like encoding and decoding within the RSU block.

            Parameters
            ----------
            x_inner : torch.Tensor
                Input tensor at the current recursion level.
            height_inner : int, optional
                Current depth level in the RSU block. Default is 1.

            Returns
            -------
            torch.Tensor
                Output tensor after passing through the nested U-structure.
            """
            if height_inner < self.height:
                x1 = getattr(self, f"rebnconv{height_inner}")(x_inner)
                if not self.dilated and height_inner < self.height - 1:
                    x2 = unet(getattr(self, "downsample")(x1), height_inner + 1)
                else:
                    x2 = unet(x1, height_inner + 1)

                x = getattr(self, f"rebnconv{height_inner}d")(torch.cat((x2, x1), dim=1))
                return _upsample_like(x, sizes[height_inner - 1]) if not self.dilated and height_inner > 1 else x
            else:
                return getattr(self, f"rebnconv{height_inner}")(x_inner)

        return x + unet(x)

    def _make_layers(self, height: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool = False) -> None:
        """
        Create and register the layers of the RSU block.

        Parameters
        ----------
        height : int
            Number of layers in the RSU block.
        in_ch : int
            Number of input channels.
        mid_ch : int
            Number of channels in the middle layers.
        out_ch : int
            Number of output channels.
        dilated : bool, optional
            Whether to use dilated convolutions. Default is False.
        """
        # Initial convolution
        self.add_module("rebnconvin", REBNCONV(in_ch, out_ch))

        # Downsampling layer
        self.add_module("downsample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

        # Encoder layers
        self.add_module(f"rebnconv1", REBNCONV(out_ch, mid_ch))
        self.add_module(f"rebnconv1d", REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f"rebnconv{i}", REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f"rebnconv{i}d", REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        # Final encoder layer
        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f"rebnconv{height}", REBNCONV(mid_ch, mid_ch, dilate=dilate))


class CustomRSU(nn.Module):
    """
    Custom Residual U-block (RSU) implemented using the AutoEncoder architecture.

    This module integrates the AutoEncoder to mimic the RSU block's functionality,
    capturing multi-scale features with optional skip connections.

    Parameters
    ----------
    autoencoder_cfg : dict
        Configuration dictionary for the AutoEncoder.
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    width : int
        Base width for the encoder and decoder layers.
    layers : Sequence[int]
        Number of blocks in each stage of the encoder and decoder.
    groups : int, optional
        Number of groups for group normalization in the block. Default is 1.
    dilation : int, optional
        Dilation rate for convolutional layers. Default is 1.
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use. Default is `nn.BatchNorm2d`.
    skip_connect : bool, optional
        If True, adds skip connections for U-Net style. Default is False.
    """

    def __init__(
        self,
        autoencoder_cfg: dict,
        ch_in: int,
        ch_out: int,
        width: int,
        layers: Sequence[int],
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        skip_connect: bool = False,
    ):
        super(CustomRSU, self).__init__()
        # Initialize the AutoEncoder as the core of RSU
        self.autoencoder = AutoEncoder(
            block=autoencoder_cfg["block"],
            ch_in=ch_in,
            ch_out=ch_out,
            width=width,
            layers=layers,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
            skip_connect=skip_connect,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Custom RSU block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ch_in, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the Custom RSU block.
        """
        return self.autoencoder(x)


class U2NET(nn.Module):
    """
    Modified U^2-Net architecture integrating Custom RSU blocks using AutoEncoder.

    This version replaces RSU blocks with CustomRSU modules based on AutoEncoder,
    enabling the use of a flexible AutoEncoder structure within U^2-Net.

    Parameters
    ----------
    cfgs : dict
        Configuration dictionary defining the structure of Custom RSU blocks and side outputs.
    out_ch : int
        Number of output channels for the final saliency map.
    autoencoder_cfg : dict
        Configuration dictionary for the AutoEncoder used in Custom RSU blocks.
    """

    def __init__(self, cfgs: dict, out_ch: int, autoencoder_cfg: dict):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self.autoencoder_cfg = autoencoder_cfg
        self._make_layers(cfgs)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass through the U^2-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        list of torch.Tensor
            List of saliency maps at different resolutions, including the final fused map.
        """
        sizes = _size_map(x, self.height)
        maps = []  # Storage for side outputs

        def unet(x_inner: torch.Tensor, height_inner: int = 1) -> torch.Tensor:
            """
            Recursive U-Net-like encoding and decoding within the U^2-Net.

            Parameters
            ----------
            x_inner : torch.Tensor
                Input tensor at the current recursion level.
            height_inner : int, optional
                Current depth level in the U^2-Net. Default is 1.
            """
            if height_inner < self.height:
                x1 = getattr(self, f"stage{height_inner}")(x_inner)
                x2 = unet(getattr(self, "downsample")(x1), height_inner + 1)
                x = getattr(self, f"stage{height_inner}d")(torch.cat((x2, x1), dim=1))
                side(x, height_inner)
                return _upsample_like(x, sizes[height_inner - 1]) if height_inner > 1 else x
            else:
                x = getattr(self, f"stage{height_inner}")(x_inner)
                side(x, height_inner)
                return _upsample_like(x, sizes[height_inner - 1])

        def side(x_side: torch.Tensor, h: int) -> None:
            """
            Generate side output saliency map before sigmoid activation.

            Parameters
            ----------
            x_side : torch.Tensor
                Tensor from the current stage for side output.
            h : int
                Current depth level in the U^2-Net.
            """
            x_side = getattr(self, f"side{h}")(x_side)
            x_side = _upsample_like(x_side, sizes[1])
            maps.append(x_side)

        def fuse() -> list:
            """
            Fuse all side outputs to generate the final saliency map.

            Returns
            -------
            list of torch.Tensor
                List of fused saliency maps after sigmoid activation.
            """
            maps.reverse()
            x_fuse = torch.cat(maps, dim=1)
            x_fuse = getattr(self, "outconv")(x_fuse)
            maps.insert(0, x_fuse)
            return [torch.sigmoid(map_i) for map_i in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs: dict) -> None:
        """
        Create and register the layers of the Modified U^2-Net.

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary defining the structure of Custom RSU blocks and side outputs.
        """
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module("downsample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # Build Custom RSU block
            self.add_module(
                k,
                CustomRSU(
                    autoencoder_cfg=self.autoencoder_cfg,
                    ch_in=v[1][1],
                    ch_out=v[1][3],
                    width=v[1][2],
                    layers=[1] * v[1][0],  # Adjust based on height
                    groups=1,  # Adjust if needed
                    dilation=v[1][4] if len(v[1]) > 4 else 1,
                    norm_layer=nn.BatchNorm2d,
                    skip_connect=False,  # Adjust based on configuration
                ),
            )
            if v[2] > 0:
                # Build side output layer
                self.add_module(f"side{v[0][-1]}", nn.Conv2d(v[2], self.out_ch, kernel_size=3, padding=1))
        # Build fuse layer
        self.add_module("outconv", nn.Conv2d(int(self.height * self.out_ch), self.out_ch, kernel_size=1))


def U2NET_full_Modified() -> U2NET:
    """
    Instantiate the full version of Modified U^2-Net with Custom RSU configurations.

    Returns
    -------
    U2NET_Modified
        The full Modified U^2-Net model instance.
    """
    full_cfg = {
        # Configuration for each stage:
        # {stage_name: [RSU name, (height, in_ch, mid_ch, out_ch, dilated), side_output_ch]}
        "stage1": ["En_1", (7, 3, 32, 64, False), -1],
        "stage2": ["En_2", (6, 64, 32, 128, False), -1],
        "stage3": ["En_3", (5, 128, 64, 256, False), -1],
        "stage4": ["En_4", (4, 256, 128, 512, False), -1],
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256, False), 256],
        "stage3d": ["De_3", (5, 512, 64, 128, False), 128],
        "stage2d": ["De_2", (6, 256, 32, 64, False), 64],
        "stage1d": ["De_1", (7, 128, 16, 64, False), 64],
    }

    # Define AutoEncoder configuration for CustomRSU
    autoencoder_cfg = {
        "block": blk.UNetBlock,  # Adjust based on your block implementation
    }

    return U2NET(cfgs=full_cfg, out_ch=1, autoencoder_cfg=autoencoder_cfg)


def U2NET_lite_Modified() -> U2NET:
    """
    Instantiate the lite version of Modified U^2-Net with simplified Custom RSU configurations.

    Returns
    -------
    U2NET_Modified
        The lite Modified U^2-Net model instance.
    """
    lite_cfg = {
        # Configuration for each stage:
        # {stage_name: [RSU name, (height, in_ch, mid_ch, out_ch, dilated), side_output_ch]}
        "stage1": ["En_1", (7, 3, 16, 64, False), -1],
        "stage2": ["En_2", (6, 64, 16, 64, False), -1],
        "stage3": ["En_3", (5, 64, 16, 64, False), -1],
        "stage4": ["En_4", (4, 64, 16, 64, False), -1],
        "stage5": ["En_5", (4, 64, 16, 64, True), -1],
        "stage6": ["En_6", (4, 64, 16, 64, True), 64],
        "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (4, 128, 16, 64, False), 64],
        "stage3d": ["De_3", (5, 128, 16, 64, False), 64],
        "stage2d": ["De_2", (6, 128, 16, 64, False), 64],
        "stage1d": ["De_1", (7, 128, 16, 64, False), 64],
    }

    # Define AutoEncoder configuration for CustomRSU
    autoencoder_cfg = {
        "block": blk.UNetBlock,  # Adjust based on your block implementation
    }

    return U2NET(cfgs=lite_cfg, out_ch=1, autoencoder_cfg=autoencoder_cfg)


# if __name__ == "__main__":
#     # 예시 입력
#     input_tensor = torch.randn(1, 3, 320, 320)  # 배치 크기 1, 채널 3, 320x320 이미지
#
#     # 전체 Modified U2-Net 모델 인스턴스 생성
#     model_full = U2NET_full_Modified()
#     print(model_full)
#
#     # 라이트 Modified U2-Net 모델 인스턴스 생성
#     model_lite = U2NET_lite_Modified()
#     print(model_lite)
#
#     # 모델을 GPU로 이동 (가능한 경우)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_full.to(device)
#     model_lite.to(device)
#     input_tensor = input_tensor.to(device)
#
#     # 모델의 forward 패스 수행
#     output_full = model_full(input_tensor)
#     output_lite = model_lite(input_tensor)
#
#     # 출력 형태 확인
#     print([o.shape for o in output_full])  # 사이드 출력 맵의 형태
#     print([o.shape for o in output_lite])
