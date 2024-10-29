from typing import OrderedDict, Tuple, Sequence, Callable

import torch
import torch.nn as nn

import math

# Import necessary modules from your project structure
from zae_engine.models import AutoEncoder
from zae_engine.models.converter import DimConverter
from zae_engine.nn_night.blocks import unet_block
from zae_engine.nn_night import blocks as blk
from zae_engine.nn_night.layers import Residual

__all__ = ["U2NET_full", "U2NET_lite"]
# https://arxiv.org/pdf/2005.09007


class U2NET(nn.Module):
    """
    U^2-Net architecture for salient object detection.

    The U^2-Net is a nested U-structure network that captures multi-scale features using RSU blocks.
    It produces side outputs at different resolutions, which are fused to generate the final saliency map.

    Parameters
    ----------
    cfgs : dict
        Configuration dictionary defining the structure of RSU blocks and side outputs.
    out_ch : int
        Number of output channels for the final saliency map.
    """

    def __init__(self, cfgs: dict, out_ch: int):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self.raw_resolution = (320, 320)  # Example resolution; adjust as needed
        self._make_layers(cfgs)

        # Initialize layers for side outputs and fusion
        self.side_layers = nn.ModuleDict()
        for k, v in cfgs.items():
            if v[2] > 0:
                self.side_layers[f"side{v[0][-1]}"] = nn.Conv2d(v[2], self.out_ch, kernel_size=3, padding=1)
        self.fuse_layer = nn.Conv2d(int((len(cfgs) + 1) / 2) * self.out_ch, self.out_ch, kernel_size=1)
        self.sig = nn.Sigmoid()

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
        maps = []  # Storage for side outputs

        # Forward through the encoder-decoder (AutoEncoder)
        x = self.encoder(x)

        # Collect side outputs
        for side_key in self.side_layers.keys():
            side_map = getattr(self, side_key)(x)
            side_map = nn.functional.interpolate(
                side_map, size=self.raw_resolution, mode="bilinear", align_corners=False
            )
            maps.append(side_map)

        # Fuse side outputs
        fused = torch.cat(maps, dim=1)
        fused = self.fuse_layer(fused)
        fused = self.sig(fused)
        maps.insert(0, fused)

        # Apply sigmoid activation to all side outputs
        maps = [self.sig(map_i) for map_i in maps]

        return maps

    def _make_layers(self, cfgs: dict) -> None:
        """
        Create and register the layers of the U^2-Net.

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary defining the structure of RSU blocks and side outputs.
        """
        self.height = int((len(cfgs) + 1) / 2)
        self.encoder = AutoEncoder(
            block=blk.UNetBlock,  # Adjust based on your block implementation
            ch_in=3,
            ch_out=64,
            width=64,
            layers=[7, 6, 5, 4, 4, 4],
            groups=1,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
            skip_connect=True,
        )
        # Assuming input resolution is known; set raw_resolution accordingly
        # self.raw_resolution = (x.shape[-2], x.shape[-1])  # Set dynamically if needed

        # Initialize side layers based on cfgs
        for k, v in cfgs.items():
            if v[2] > 0:
                # Side output layers are already handled in self.side_layers
                pass

        # Fuse layer is already initialized in __init__


def U2NET_full() -> U2NET:
    """
    Instantiate the full version of U^2-Net with detailed RSU configurations.

    Returns
    -------
    U2NET
        The full U^2-Net model instance.
    """
    full_cfg = {
        # Configuration for each stage: {stage_name: [RSU name, (height, in_ch, mid_ch, out_ch, dilated), side_output_ch]}
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
    return U2NET(cfgs=full_cfg, out_ch=1)


def U2NET_lite() -> U2NET:
    """
    Instantiate the lite version of U^2-Net with simplified RSU configurations.

    Returns
    -------
    U2NET
        The lite U^2-Net model instance.
    """
    lite_cfg = {
        # Configuration for each stage: {stage_name: [RSU name, (height, in_ch, mid_ch, out_ch, dilated), side_output_ch]}
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
    return U2NET(cfgs=lite_cfg, out_ch=1)


class MyU2NET(AutoEncoder):
    """
    Custom U2-Net implementation based on AutoEncoder with integrated RSU blocks.

    Parameters
    ----------
    ch_in : int, optional
        Number of input channels. Default is 3.
    ch_out : int, optional
        Number of output channels. Default is 1.
    width : int, optional
        Base width for the encoder and decoder layers. Default is 16.
    heights : Sequence[int], optional
        Heights for each RSU block. Default is (7, 6, 5, 4, 4, 4).
    dilation_heights : Sequence[int], optional
        Dilation settings for each RSU block. Default is (7, 6, 5, 4, 1, 1).
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use. Default is `nn.BatchNorm2d`.
    """

    def __init__(
        self,
        ch_in: int = 3,
        ch_out: int = 1,
        width: int = 64,
        heights: Sequence[int] = (7, 6, 5, 4, 4, 4),
        dilation_heights: Sequence[int] = (7, 6, 5, 4, -1, -1),
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ):
        self.ch_in = ch_in
        self.ch_out = ch_out
        super(MyU2NET, self).__init__(
            block=blk.UNetBlock,
            ch_in=ch_in,
            ch_out=ch_out,
            width=width,
            layers=[1] * len(heights),  # Internal layers will be replaced below
            norm_layer=norm_layer,
            skip_connect=True,
        )
        self.main_height = len(heights)
        self.side_outputs = []  # Storage for side outputs

        # Remove encoder and bottleneck hooks to prevent feature_vectors accumulation
        self.remove_hooks()

        # Initialize RSU blocks based on heights and dilation_heights
        ch_ = width
        for i, (h, dh) in enumerate(zip(heights, dilation_heights)):
            # Configure AutoEncoder parameters based on height and dilation
            rsu_cfg = {
                "block": blk.UNetBlock,
                "width": ch_,
                "layers": [2] + [1] * (h - 1),
                "norm_layer": norm_layer,
                "skip_connect": True,
            }
            # Initialize encoder and decoder RSU blocks with residual connections
            u_encoder = AutoEncoder(ch_in=ch_ if i else ch_in, ch_out=ch_ * 2, **rsu_cfg)
            u_decoder = AutoEncoder(ch_in=ch_ * 2, ch_out=ch_, **rsu_cfg)

            # # todo: h와 dh 비교로 dilation 범위 설정해야 함
            # # Adjust dilation and downsampling based on dilation_heights
            # for ii in range(dh, h):
            #     # Access the appropriate convolutional layer to set dilation
            #     # 여기서는 'conv_s1'이라는 레이어 이름을 가정
            #     # 실제 레이어 이름에 맞게 수정해야 합니다.
            #     target_layer = u_encoder.encoder.body[ii][0].conv1
            #     target_const = DimConverter.const_getter(target_layer)
            #     target_const["dilation"] = (2,)
            #     u_encoder.encoder.body[ii][0].conv1 = type(target_layer)(**target_const)
            #
            #     target_layer = u_encoder.encoder.body[ii][0].conv2
            #     target_const = DimConverter.const_getter(target_layer)
            #     target_const["dilation"] = (2,)
            #     u_encoder.encoder.body[ii][0].conv2 = type(target_layer)(**target_const)
            #
            #
            #     # Replace downsample with identity if dilation is applied
            #     u_encoder.encoder.body[ii].downsample = nn.Identity()
            #     u_decoder.encoder.body[ii].downsample = nn.Identity()

            # Replace fc layer with identity
            u_encoder.fc = nn.Identity()
            u_decoder.fc = nn.Identity()

            # Replace the corresponding encoder and decoder blocks
            self.encoder.body[i] = Residual(u_encoder)
            self.decoder[i] = Residual(u_decoder)

            if dh:
                ch_ *= 2

        # Register hooks only for decoder blocks to capture side outputs
        for decoder_block in self.decoder:
            decoder_block.register_forward_hook(self.decoding_output_hook)

            # 퓨전 레이어 교체 (이미 self.fc가 정의되어 있으므로 필요 시 수정)
            self.fc = nn.Conv2d(in_channels=width, out_channels=ch_out, kernel_size=1)
            self.sig = nn.Sigmoid()

    def decoding_output_hook(self, module, input_tensor, output_tensor):
        """
        Hook to capture decoder outputs for side outputs.
        """
        self.side_outputs.append(output_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MyU2NET model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape: (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The final saliency map. Shape: (batch_size, ch_out, height, width).
        """
        # Clear previous side outputs and feature_vectors
        self.side_outputs = []
        self.feature_vectors = []

        # Forward pass through encoder and bottleneck
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)

        # Forward pass through decoder with skip connections
        for up_pool, dec in zip(self.up_pools, self.decoder):
            bottleneck = up_pool(bottleneck)
            if self.skip_connect and len(self.feature_vectors) > 0:
                bottleneck = torch.cat((bottleneck, self.feature_vectors.pop()), dim=1)
            bottleneck = dec(bottleneck)

        # Apply fully-connected layer and sigmoid activation
        output = self.sig(self.fc(bottleneck))

        # Fuse side outputs
        if self.side_outputs:
            fused = torch.zeros_like(output)
            for side in self.side_outputs:
                side = nn.functional.interpolate(side, size=output.shape[-2:], mode="bilinear", align_corners=False)
                fused += side
            fused = self.sig(self.fc(fused))
            output = fused  # Replace output with fused output

        return output


class U2Net(AutoEncoder):
    """
    U²-Net model built using the AutoEncoder framework with RSU7 blocks.
    """

    def __init__(self, in_ch=1, out_ch=1, width=4, layers=[2, 2, 2, 2, 2, 2]):
        """
        Initializes the U²-Net model using AutoEncoder.

        Parameters
        ----------
        in_ch : int, optional
            Number of input channels, by default 1.
        out_ch : int, optional
            Number of output channels, by default 1.
        width : int, optional
            Base width for the encoder and decoder layers, by default 32.
        layers : Sequence[int], optional
            Number of RSU7 blocks in each stage, by default [2, 2, 2, 2, 2, 2].
        """
        super(U2Net, self).__init__(
            block=unet_block.RSU7,
            ch_in=in_ch,
            ch_out=out_ch,
            width=width,
            layers=layers,
            groups=1,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
            skip_connect=True,
        )
        print("U2NetAutoEncoder initialized with RSU7 blocks.")

    def forward(self, x):
        """
        Forward pass for U2NetAutoEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape: (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor. Shape: (batch_size, channels, height, width).
        """
        return super(U2Net, self).forward(x)


if __name__ == "__main__":
    # 예시 입력
    input_tensor = torch.randn(1, 3, 320, 320)  # 배치 크기 1, 채널 3, 320x320 이미지

    # 전체 U2-Net 모델 인스턴스 생성
    model_full = U2Net()
    print(model_full)

    # 모델을 GPU로 이동 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_full.to(device)
    input_tensor = input_tensor.to(device)

    # 모델의 forward 패스 수행
    output_full = model_full(input_tensor)

    # 출력 형태 확인
    print([o.shape for o in output_full])  # 사이드 출력 맵의 형태
    print("done")
