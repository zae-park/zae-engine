from typing import OrderedDict, Union, Sequence, Callable

import torch
import torch.nn as nn

import math

# Import necessary modules from your project structure
from zae_engine.models import AutoEncoder
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


if __name__ == "__main__":
    # 예시 입력
    input_tensor = torch.randn(1, 3, 320, 320)  # 배치 크기 1, 채널 3, 320x320 이미지

    # 전체 U2-Net 모델 인스턴스 생성
    model_full = U2NET_full()
    print(model_full)

    # 라이트 U2-Net 모델 인스턴스 생성
    model_lite = U2NET_lite()
    print(model_lite)

    # 모델을 GPU로 이동 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_full.to(device)
    model_lite.to(device)
    input_tensor = input_tensor.to(device)

    # 모델의 forward 패스 수행
    output_full = model_full(input_tensor)
    output_lite = model_lite(input_tensor)

    # 출력 형태 확인
    print([o.shape for o in output_full])  # 사이드 출력 맵의 형태
    print([o.shape for o in output_lite])
