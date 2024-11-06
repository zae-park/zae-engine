import math
from functools import partial
from typing import OrderedDict, Union, Sequence, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Import necessary modules from your project structure
from .autoencoder import AutoEncoder
from ...nn_night.blocks.unet_block import RSUBlock
from ...nn_night import blocks as blk
from ...nn_night.layers import Residual

__all__ = ["NestedUNet"]
# https://arxiv.org/pdf/2005.09007


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


class NestedUNet(nn.Module):
    """
    U²-Net 구조 구현.

    Parameters
    ----------
    in_ch : int, optional
        입력 채널 수. Default is 3.
    out_ch : int, optional
        출력 채널 수. Default is 1.
    width : int, optional
        초기 중간 채널 수. Default is 32.
    heights : List[int]
        각 인코더 레이어의 RSU 블록 높이 리스트.
    dilation_heights : List[int]
        각 인코더 레이어의 RSU 블록 dilation 높이 리스트.
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        width: Union[int | Sequence] = 32,
        heights: Sequence[int] = (7, 6, 5, 4, 4),
        dilation_heights: Sequence[int] = (2, 2, 2, 2, 4),
        middle_width: Union[int | Sequence] = (32, 32, 64, 128, 256),
    ):
        super(NestedUNet, self).__init__()

        assert len(heights) == len(dilation_heights), "heights and dilation_heights must have the same length."

        self.minimum_resolution = max([2 ** (h - 1 + i) for i, h in enumerate(heights)])
        self.num_layers = len(heights)

        # 인코더 설정
        self.encoder = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.encoder_channels = []

        # 입력 width 확인
        if isinstance(width, int):
            width_list = [width * 2**i for i in range(len(heights))]
        else:
            assert len(width) == len(heights)
            width_list = width

        for i, (h, dh, w, mw) in enumerate(zip(heights, dilation_heights, width_list, middle_width)):

            enc_ch_in = w if i else in_ch
            enc_ch_out = w if h == dh else 2 * w
            self.encoder.append(RSUBlock(ch_in=enc_ch_in, ch_mid=mw, ch_out=enc_ch_out, height=h, dilation_height=dh))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            print(f"\tEnc_{i}\tch_in: {enc_ch_in}\tch_mid: {mw}\tch_out: {enc_ch_out}\th: {h}\tdh: {dh}")

            #
            # # 다음 인코더 레이어를 위해 채널 설정
            # self.encoder_channels.append((current_in_ch, current_mid_ch, out_ch_enc))
            # current_in_ch = out_ch_enc
            # current_mid_ch = out_ch_enc  # 다음 레이어의 mid_ch는 현재 레이어의 out_ch

        # 병목
        bottleneck_height = heights[-1]  # 병목도 동일한 height 사용
        bottleneck_dil = dilation_heights[-1]
        self.bottleneck = RSUBlock(
            ch_in=enc_ch_out,
            ch_mid=mw,
            ch_out=enc_ch_out,
            height=bottleneck_height,
            dilation_height=bottleneck_dil,
        )
        print(
            f"\tBottle\tch_in: {enc_ch_out}\tch_mid: {mw}\tch_out: {enc_ch_out}\th: {bottleneck_height}\tdh: {bottleneck_dil}"
        )

        # 디코더 설정
        self.up_layers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.decoder_channels = []

        for i, (h, dh, w, mw) in enumerate(
            zip(heights[::-1], dilation_heights[::-1], width_list[::-1], middle_width[::-1])
        ):
            dec_ch_out = 2 * w if h == dh else w // 2

            self.up_layers.append(nn.Upsample(scale_factor=2))

            self.decoder.append(RSUBlock(ch_in=2 * w, ch_mid=mw, ch_out=dec_ch_out, height=h, dilation_height=dh))
            print(f"\tDec_{i}\tch_in: {2 * w}\tch_mid: {mw}\tch_out: {dec_ch_out}\th: {h}\tdh: {dh}")
            self.decoder_channels.append((2 * w, mw, dec_ch_out))

        # 출력 레이어
        self.out_conv = nn.Conv2d(dec_ch_out, out_ch, kernel_size=1)

        # 사이드 아웃풋 (딥 슈퍼비전을 위한 추가적인 출력)
        self.side_layers = nn.ModuleList()
        for i in range(self.num_layers):
            # 각 디코더 단계마다 사이드 아웃풋을 생성
            self.side_layers.append(nn.Conv2d(self.decoder_channels[i][2], out_ch, kernel_size=3, padding=1))

    def forward(self, x):
        encoder_features = []

        # 인코더 경로
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            encoder_features.append(x)
            x = self.pool_layers[i](x)

        # 병목
        x = self.bottleneck(x)

        side_outputs = []

        # 디코더 경로
        for i in range(self.num_layers):
            x = self.up_layers[i](x)
            enc_feat = encoder_features[self.num_layers - i - 1]

            # # 크기가 다를 경우 맞추기
            # if x.shape[2:] != enc_feat.shape[2:]:
            #     x = F.interpolate(x, size=enc_feat.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder[i](x)

            # 사이드 아웃풋 생성
            side_output = self.side_layers[i](x)
            side_outputs.append(side_output)

        # 최종 출력
        out = self.out_conv(x)

        return out, *side_outputs


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
            # todo: use RSUBlock instead
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
