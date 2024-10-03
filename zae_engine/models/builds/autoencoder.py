from typing import Callable, Tuple, Type, Union, List
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from . import cnn
from ...nn_night import blocks as blk


class AutoEncoder(nn.Module):
    """
    A flexible AutoEncoder architecture with optional skip connections for U-Net style implementations.

    Parameters
    ----------
    block : Type[Union[blk.UNetBlock, nn.Module]]
        The basic building block for the encoder and decoder (e.g., ResNet block or UNetBlock).
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    width : int
        Base width for the encoder and decoder layers.
    layers : Union[Tuple[int], List[int]]
        Number of blocks in each stage of the encoder and decoder.
    groups : int, optional
        Number of groups for group normalization in the block. Default is 1.
    dilation : int, optional
        Dilation rate for convolutional layers. Default is 1.
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use. Default is `nn.BatchNorm2d`.
    skip_connect : bool, optional
        If True, adds skip connections for U-Net style. Default is False.

    Attributes
    ----------
    encoder : nn.Module
        The encoder module that encodes the input image.
    bottleneck : nn.Module
        The bottleneck layer between the encoder and decoder.
    decoder : nn.ModuleList
        The decoder module that reconstructs the input image.
    feature_vectors : list
        Stores intermediate feature maps for skip connections when `skip_connect` is True.
    up_pools : nn.ModuleList
        List of transposed convolution layers for upsampling in the decoder.
    fc : nn.Conv2d
        The final output convolutional layer.
    sig : nn.Sigmoid
        Sigmoid activation function for the output.

    Methods
    -------
    feature_hook(module, input_tensor, output_tensor)
        Hooks intermediate feature maps for skip connections.
    feature_output_hook(module, input_tensor, output_tensor)
        Hooks the final feature map before bottleneck.
    forward(x)
        Defines the forward pass of the autoencoder.
    """

    def __init__(
        self,
        block: Type[Union[blk.UNetBlock, nn.Module]],
        ch_in: int,
        ch_out: int,
        width: int,
        layers: Union[Tuple[int], List[int]],
        groups: int = 1,
        dilation: int = 1,
        # zero_init_residual: bool = False,
        # replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        skip_connect: bool = False,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder = cnn.CNNBase(
            block=block,
            ch_in=ch_in,
            ch_out=ch_out,
            width=width,
            layers=layers,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        
        self.skip_connect = skip_connect
        self.encoder.stem = nn.Identity()
        self.encoder.body[0] = self.encoder.make_body(blocks=[block] * layers[0], ch_in=ch_in, ch_out=width, stride=2)

        self.feature_vectors = []
        # [U-net] Register hook for every blocks in encoder when "skip_connect" is true.
        if skip_connect:
            for b in self.encoder.body:
                b[0].relu2.register_forward_hook(self.feature_hook)
        # registrate hook to end of last body instead of pooling layer
        self.encoder.body[-1].register_forward_hook(self.feature_output_hook)

        self.bottleneck = block(width * 8, width * 16)

        up_pools = []
        decoder = []
        for i, l in enumerate(layers):
            c_i, c_o = width * 2 ** (i + 1), width * 2**i
            up_pools.append(nn.ConvTranspose2d(in_channels=c_i, out_channels=c_o, kernel_size=2, stride=2))
            decoder.append(self.encoder.make_body([block] * l, ch_in=c_i if skip_connect else c_i // 2, ch_out=c_o))
        self.up_pools = nn.ModuleList(reversed(up_pools))
        self.decoder = nn.ModuleList(reversed(decoder))

        self.fc = nn.Conv2d(in_channels=width, out_channels=ch_out, kernel_size=1)
        self.sig = nn.Sigmoid()

    def feature_hook(self, module, input_tensor, output_tensor):
        """
        Hooks intermediate feature maps for skip connections.
        """
        self.feature_vectors.append(input_tensor[0])

    def feature_output_hook(self, module, input_tensor, output_tensor):
        """
        Hooks the final feature map before bottleneck.
        """
        self.feature_vectors.append(output_tensor)

    def forward(self, x):
        """
        Defines the forward pass of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape: (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The reconstructed output tensor. Shape: (batch_size, channels, height, width).
        """
        # Ensure feature_vectors is cleared at the start of each forward pass
        self.feature_vectors = []

        # Forwarding encoder & hook immediate outputs
        feat = self.encoder(x)

        # Bottleneck processing
        feat = self.bottleneck(self.feature_vectors.pop())

        # Decoder with skip connections if enabled
        for up_pool, dec in zip(self.up_pools, self.decoder):
            feat = up_pool(feat)
            if self.skip_connect and len(self.feature_vectors) > 0:
                feat = torch.cat((feat, self.feature_vectors.pop()), dim=1)
            feat = dec(feat)

        return self.sig(self.fc(feat))


class VAE(AutoEncoder):
    """
    Variational AutoEncoder (VAE) architecture extending AutoEncoder.

    Parameters
    ----------
    block : Type[Union[blk.UNetBlock, nn.Module]]
        The basic building block for the encoder and decoder (e.g., ResNet block or UNetBlock).
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    width : int
        Base width for the encoder and decoder layers.
    layers : Union[Tuple[int], List[int]]
        Number of blocks in each stage of the encoder and decoder.
    groups : int, optional
        Number of groups for group normalization in the block. Default is 1.
    dilation : int, optional
        Dilation rate for convolutional layers. Default is 1.
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use. Default is `nn.BatchNorm2d`.
    skip_connect : bool, optional
        If True, adds skip connections for U-Net style. Default is False.
    latent_dim : int, optional
        Dimension of the latent space. Default is 128.
    encoder_output_shape : List[int]
        The shape of the encoder's output (excluding batch size), e.g., [channels, height, width].

    Attributes
    ----------
    encoder : nn.Module
        The encoder module that encodes the input image.
    bottleneck : nn.Module
        The bottleneck layer between the encoder and decoder.
    decoder : nn.ModuleList
        The decoder module that reconstructs the input image.
    feature_vectors : list
        Stores intermediate feature maps for skip connections when `skip_connect` is True.
    up_pools : nn.ModuleList
        List of transposed convolution layers for upsampling in the decoder.
    fc : nn.Conv2d
        The final output convolutional layer.
    sig : nn.Sigmoid
        Sigmoid activation function for the output.
    fc_mu : nn.Linear
        Fully connected layer to generate mean of latent distribution.
    fc_logvar : nn.Linear
        Fully connected layer to generate log variance of latent distribution.
    fc_z : nn.Linear
        Fully connected layer to map sampled latent variable back to encoder channels.
    encoder_output_shape : List[int]
        The shape of the encoder's output (excluding batch size), e.g., [channels, height, width].
    encoder_output_features : int
        Total number of features after flattening the encoder's output.
    """

    def __init__(
        self,
        block: Type[Union[blk.UNetBlock, nn.Module]],
        ch_in: int,
        ch_out: int,
        width: int,
        layers: Union[Tuple[int], List[int]],
        encoder_output_shape: List[int],
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        skip_connect: bool = False,
        latent_dim: int = 128,
    ):
        super(VAE, self).__init__(
            block=block,
            ch_in=ch_in,
            ch_out=ch_out,
            width=width,
            layers=layers,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
            skip_connect=skip_connect,
        )
        self.latent_dim = latent_dim
        self.encoder_output_shape = encoder_output_shape

        # Calculate the total number of features from encoder_output_shape
        self.encoder_output_features = 1
        for dim in self.encoder_output_shape:
            self.encoder_output_features *= dim

        # Define Linear layers to generate mu and logvar
        self.fc_mu = nn.Linear(self.encoder_output_features, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_features, latent_dim)

        # Define Linear layer to map latent variable back to encoder channels
        self.fc_z = nn.Linear(latent_dim, self.encoder_output_features)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.
        logvar : torch.Tensor
            Log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent variable z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Defines the forward pass of the VAE.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape: (batch_size, channels, height, width).

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            The reconstructed output tensor, along with mu and logvar.
        """
        # Ensure feature_vectors is cleared at the start of each forward pass
        self.feature_vectors = []

        # Forward encoder
        _ = self.encoder(x)
        feat = self.feature_vectors.pop()

        # Flatten the encoder output
        feat_flat = feat.view(feat.size(0), -1)

        # Generate mu and logvar using Linear layers
        mu = self.fc_mu(feat_flat)
        logvar = self.fc_logvar(feat_flat)

        # Reparameterize to obtain latent variable z
        z = self.reparameterize(mu, logvar)

        # Map z back to feature space using Linear layer
        z_mapped = self.fc_z(z)

        # Reshape z_mapped to match encoder_output_shape
        # Calculate the shape excluding the batch dimension
        z_shape = [feat.size(0)] + self.encoder_output_shape
        z_reshaped = z_mapped.view(*z_shape)

        # Bottleneck processing
        feat = self.bottleneck(z_reshaped)

        # Decoder with skip connections if enabled
        for up_pool, dec in zip(self.up_pools, self.decoder):
            feat = up_pool(feat)
            if self.skip_connect and len(self.feature_vectors) > 0:
                feat = torch.cat((feat, self.feature_vectors.pop()), dim=1)
            feat = dec(feat)

        # Final output
        reconstructed = self.sig(self.fc(feat))
        return reconstructed, mu, logvar
