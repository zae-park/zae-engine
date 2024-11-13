from typing import OrderedDict, Union, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn_night.blocks.unet_block import RSUBlock

__all__ = ["NestedUNet"]
# https://arxiv.org/pdf/2005.09007


class NestedUNet(nn.Module):
    """
    Implementation of the U²-Net architecture.

    Parameters
    ----------
    in_ch : int, optional
        Number of input channels. Default is 3.
    out_ch : int, optional
        Number of output channels. Default is 1.
    width : Union[int, Sequence], optional
        Initial number of middle channels. Default is 32.
    heights : Sequence[int], optional
        List of RSU block heights for each encoder layer. Default is (7, 6, 5, 4, 4).
    dilation_heights : Sequence[int], optional
        List of dilation heights for each encoder layer. Default is (2, 2, 2, 2, 4).
    middle_width : Union[int, Sequence], optional
        List of middle channels for each RSU block. Default is (32, 32, 64, 128, 256).

    References
    ----------
    .. [1] Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020).
            U2-Net: Going deeper with nested U-structure for salient object detection. Pattern recognition, 106, 107404.
            (https://arxiv.org/pdf/2005.09007)
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        width: Union[int, Sequence] = 32,
        heights: Sequence[int] = (7, 6, 5, 4, 4),
        dilation_heights: Sequence[int] = (2, 2, 2, 2, 4),
        middle_width: Union[int, Sequence] = (32, 32, 64, 128, 256),
    ):
        super(NestedUNet, self).__init__()

        assert len(heights) == len(dilation_heights), "heights and dilation_heights must have the same length."

        self.minimum_resolution = max([2 ** (h - 1 + i) for i, h in enumerate(heights)])
        self.num_layers = len(heights)

        # Encoder configuration
        self.encoder = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.encoder_channels = []

        # Verify input width
        if isinstance(width, int):
            width_list = [width * 2**i for i in range(len(heights))]
        else:
            assert len(width) == len(heights)
            width_list = width

        for i, (h, dh, w, mw) in enumerate(zip(heights, dilation_heights, width_list, middle_width)):

            enc_ch_in = w if i else in_ch
            enc_ch_out = 2 * w  # Double the channels
            self.encoder.append(RSUBlock(ch_in=enc_ch_in, ch_mid=mw, ch_out=enc_ch_out, height=h, dilation_height=dh))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # print(f"\tEnc_{i}\tch_in: {enc_ch_in}\tch_mid: {mw}\tch_out: {enc_ch_out}\th: {h}\tdh: {dh}")

        # Bottleneck
        bottleneck_height = heights[-1]  # Use the same height as the last encoder layer
        bottleneck_dil = dilation_heights[-1]
        self.bottleneck = RSUBlock(
            ch_in=enc_ch_out,
            ch_mid=mw,
            ch_out=enc_ch_out,
            height=bottleneck_height,
            dilation_height=bottleneck_dil,
        )
        # print(
        #     f"\tBottle\tch_in: {enc_ch_out}\tch_mid: {mw}\tch_out: {enc_ch_out}\th: {bottleneck_height}\tdh: {bottleneck_dil}"
        # )

        # Decoder configuration
        self.up_layers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.decoder_channels = []

        dec_ch_ = enc_ch_out * 2
        for i, (h, dh, w, mw) in enumerate(
            zip(heights[::-1], dilation_heights[::-1], width_list[::-1], middle_width[::-1])
        ):
            dec_ch_out = w  # Set output channels
            self.up_layers.append(nn.Upsample(scale_factor=2))

            self.decoder.append(
                RSUBlock(
                    ch_in=dec_ch_,  # Concatenated channels from skip connection
                    ch_mid=mw,
                    ch_out=dec_ch_out,
                    height=h,
                    dilation_height=dh,
                )
            )
            # print(f"\tDec_{i}\tch_in: {dec_ch_}\tch_mid: {mw}\tch_out: {dec_ch_out}\th: {h}\tdh: {dh}")
            self.decoder_channels.append((dec_ch_, mw, dec_ch_out))
            dec_ch_ = 2 * dec_ch_out

        # Output layer
        self.out_conv = nn.Conv2d(dec_ch_out, out_ch, kernel_size=1)

        # Side outputs for deep supervision
        self.side_layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Create side outputs for each decoder stage
            self.side_layers.append(nn.Conv2d(self.decoder_channels[i][2], out_ch, kernel_size=3, padding=1))

    def forward(self, x):
        encoder_features = []

        # Encoder path
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            encoder_features.append(x)
            x = self.pool_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        side_outputs = []

        # Decoder path
        for i in range(self.num_layers):
            x = self.up_layers[i](x)
            enc_feat = encoder_features[self.num_layers - i - 1]

            # Resize if necessary
            # if x.shape[2:] != enc_feat.shape[2:]:
            #     x = F.interpolate(x, size=enc_feat.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder[i](x)

            # Generate side output
            side_output = self.side_layers[i](x)
            side_outputs.append(side_output)

        # Final output
        out = self.out_conv(x)

        return out, *side_outputs
