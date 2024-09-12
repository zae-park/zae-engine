from typing import Type, TypeVar, Union

import torch
import torch.nn as nn


# TODO: implement Scaled-Dot Product Attention (SDPA).
# ref: https://magentino.tistory.com/176
# ref: https://tutorials.pytorch.kr/intermediate/scaled_dot_product_attention_tutorial.html
# from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


class TransformerBase(nn.Module):
    """
    A flexible Transformer model that supports both encoder-only and encoder-decoder architectures.

    Parameters
    ----------
    encoder_embedding : nn.Module
        The embedding layer for the encoder input.
    decoder_embedding : nn.Module, optional
        The embedding layer for the decoder input. If not provided, encoder_embedding is used for both encoder and decoder.
    encoder : nn.Module, optional
        The encoder module. Defaults to nn.Identity(), which can be replaced with any custom encoder (e.g., TransformerEncoder).
    decoder : nn.Module, optional
        The decoder module. If None, the model operates as an encoder-only model (e.g., BERT). Otherwise, uses a decoder (e.g., for translation models).

    Notes
    -----
    - If `decoder` is None, the model acts as an encoder-only transformer (similar to BERT).
    - If `decoder` is provided, the model functions as an encoder-decoder transformer (e.g., for translation tasks).
    - The forward pass adjusts based on the presence of the decoder.

    Methods
    -------
    forward(src, tgt=None, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None)
        Forward pass through the model. If `tgt` and `decoder` are provided, both encoder and decoder are used. Otherwise, only the encoder is applied.

    """

    def __init__(
        self,
        encoder_embedding: nn.Module,
        decoder_embedding: nn.Module = None,
        encoder: nn.Module = nn.Identity(),
        decoder: nn.Module = None,  # Set decoder to None by default
    ):
        super().__init__()
        self.encoder_embedding = encoder_embedding
        # If no decoder_embedding is provided, use encoder_embedding for both
        self.decoder_embedding = decoder_embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src, tgt=None, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None
    ):
        """
        Forward pass through the Transformer model.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor representing the source sequence (e.g., for BERT-style models). Shape: (batch_size, seq_len).
        tgt : torch.Tensor, optional
            The input tensor representing the target sequence (for models with a decoder). Shape: (batch_size, seq_len).
        src_mask : torch.Tensor, optional
            Source mask for masking certain positions in the encoder input.
        tgt_mask : torch.Tensor, optional
            Target mask for masking certain positions in the decoder input.
        src_key_padding_mask : torch.Tensor, optional
            Mask for padding tokens in the source sequence.
        tgt_key_padding_mask : torch.Tensor, optional
            Mask for padding tokens in the target sequence.

        Returns
        -------
        torch.Tensor
            If a decoder is provided, returns the output of the decoder. Otherwise, returns the output of the encoder.
        """
        # Apply embeddings to source and target sequences
        src_embed = self.encoder_embedding(src)

        # If a decoder exists, apply decoder embedding and pass through the decoder
        if self.decoder is not None and tgt is not None:
            if self.decoder_embedding is not None:
                tgt_embed = self.decoder_embedding(tgt)
            else:
                tgt_embed = self.encoder_embedding(tgt)
            encoded = self.encoder(src_embed, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            out = self.decoder(
                tgt_embed,
                encoded,
                tgt_mask=tgt_mask,
                memory_mask=src_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            return out
        else:
            # If no decoder, only pass through the encoder
            return self.encoder(src_embed, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


class BertBase(TransformerBase):
    """
    BertBase is a specialized version of TransformerBase, including a pooler for processing the [CLS] token.

    This class adds a pooler layer that processes the first token ([CLS]) from the encoder output, similar to the
    original BERT architecture. If a hidden dimension is provided during initialization, the pooler will be applied.
    Otherwise, only the encoder output is returned.

    Parameters
    ----------
    encoder_embedding : nn.Module
        The embedding layer for the encoder input.
    encoder : nn.Module
        The encoder module responsible for transforming the input sequence.
    dim_hidden : int, optional
        The hidden dimension used by the pooler layer. If provided, a pooler layer will be applied to the [CLS] token
        (first token) of the encoder output. Otherwise, the encoder output is returned without pooling.

    Methods
    -------
    forward(input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, src_mask=None, src_key_padding_mask=None, past_key_values_length=0)
        Performs the forward pass. If a hidden dimension (dim_hidden) is provided, the pooler is applied to the
        [CLS] token. Otherwise, it returns the encoder output as-is.
    """

    def __init__(self, encoder_embedding: nn.Module, encoder: nn.Module, **kwargs):
        super().__init__(encoder_embedding=encoder_embedding, encoder=encoder, decoder=None)
        self.dim_hidden = kwargs.get("dim_hidden", None)
        if self.dim_hidden:
            self.pool_dense = nn.Linear(self.dim_hidden, self.dim_hidden)
            self.pool_activation = nn.Tanh()

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        src_mask=None,
        src_key_padding_mask=None,
        past_key_values_length=0,
    ):
        """
        Forward pass through the BERT model with an optional pooler.

        If a hidden dimension is provided, the pooler is applied to the first token of the encoder output.
        Otherwise, the encoder output is returned as-is.

        Parameters
        ----------
        input_ids : torch.Tensor, optional
            The input tensor representing the token indices. Shape: (batch_size, seq_len).
        position_ids : torch.Tensor, optional
            The tensor representing the positions of the tokens in the sequence. Shape: (batch_size, seq_len).
        token_type_ids : torch.Tensor, optional
            The tensor representing segment IDs (e.g., sentence A or B). Shape: (batch_size, seq_len).
        inputs_embeds : torch.Tensor, optional
            Optionally, instead of input_ids, directly pass in embeddings. Shape: (batch_size, seq_len, embedding_dim).
        src_mask : torch.Tensor, optional
            Source mask for masking certain positions in the encoder input. Shape: (batch_size, seq_len).
        src_key_padding_mask : torch.Tensor, optional
            Mask for padding tokens in the source sequence. Shape: (batch_size, seq_len).
        past_key_values_length : int, optional
            If using caching, this represents the length of previously cached tokens.

        Returns
        -------
        torch.Tensor
            If dim_hidden is provided, returns the pooled output from the [CLS] token. Otherwise, returns the
            encoder output for the entire sequence. Shape: (batch_size, dim_hidden) if pooled, or
            (batch_size, seq_len, dim_hidden) if not.
        """
        if input_ids is not None:
            src = input_ids
        elif inputs_embeds is not None:
            src = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        # Apply the embedding layer
        embedding_output = self.encoder_embedding(src)

        # Add position_ids and token_type_ids if provided
        if position_ids is not None or token_type_ids is not None:
            position_embeds = self.encoder_embedding[1](position_ids)
            token_type_embeds = self.encoder_embedding[2](token_type_ids)
            embedding_output += position_embeds + token_type_embeds

        # Encode the input
        encoded_output = self.encoder(embedding_output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Apply pooler if dim_hidden is set
        if self.dim_hidden:
            cls_tkn = encoded_output[:, 0]  # Extract the first token ([CLS] token)
            return self.pool_activation(self.pool_dense(cls_tkn))
        return encoded_output


class CoderBase(nn.Module):
    """
    Base class for both Encoder and Decoder that defines the core structure of the transformer layers.

    Parameters
    ----------
    d_model : int
        The dimension of the embedding space (output size of each layer).
    num_layers : int
        The number of layers in the encoder/decoder.
    layer_factory : nn.Module, optional
        Custom layer module. Defaults to `nn.TransformerEncoderLayer` for encoders and `nn.TransformerDecoderLayer` for decoders.
    # norm_layer : str or nn.Module, optional
    #     The normalization layer to apply. Can be a string or custom `nn.Module`. Default is 'LayerNorm'.
    dim_feedforward : int, optional
        The dimension of the feedforward network. Default is 2048.
    dropout : float, optional
        Dropout rate for regularization. Default is 0.1.
    num_heads : int, optional
        Number of attention heads in multi-head attention. Default is 8.
    factory_kwargs : dict, optional
        Additional arguments to pass to `layer_factory` when creating layers.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        layer_factory: Type[nn.Module] = nn.TransformerEncoderLayer,
        # norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        super(CoderBase, self).__init__()
        self.d_model = d_model

        # Create layers using the provided layer factory
        self.layers = nn.ModuleList(
            [
                layer_factory(
                    d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, **factory_kwargs
                )
                for _ in range(num_layers)
            ]
        )

    def _get_norm_layer(self, norm_type):
        """
        Returns the appropriate normalization layer based on user input.
        """
        if isinstance(norm_type, nn.Module):
            return norm_type
        elif norm_type == "LayerNorm":
            return nn.LayerNorm(self.d_model)
        elif norm_type == "BatchNorm1d":
            return nn.BatchNorm1d(self.d_model)
        elif norm_type == "InstanceNorm1d":
            return nn.InstanceNorm1d(self.d_model)
        elif norm_type == "GroupNorm":
            return nn.GroupNorm(8, self.d_model)
        else:
            raise ValueError(f"Unsupported norm layer type: {norm_type}")


class EncoderBase(CoderBase):
    """
    Encoder class that builds on CoderBase for encoding the input sequences.

    Parameters
    ----------
    d_model : int
        The dimension of the embedding space (output size of each layer).
    num_layers : int
        The number of layers in the encoder.
    layer_factory : nn.Module, optional
        Custom layer module. Defaults to `nn.TransformerEncoderLayer`.
    # norm_layer : str or nn.Module, optional
    #     The normalization layer to apply. Can be a string or custom `nn.Module`. Default is 'LayerNorm'.
    dim_feedforward : int, optional
        The dimension of the feedforward network. Default is 2048.
    dropout : float, optional
        Dropout rate for regularization. Default is 0.1.
    num_heads : int, optional
        Number of attention heads in multi-head attention. Default is 8.
    factory_kwargs : dict, optional
        Additional arguments to pass to `layer_factory` when creating layers.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        layer_factory: Type[nn.Module] = nn.TransformerEncoderLayer,
        # norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        super(EncoderBase, self).__init__(
            d_model, num_layers, layer_factory, dim_feedforward, dropout, num_heads, **factory_kwargs
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor representing the source sequence. Shape: (batch_size, seq_len, d_model).
        src_mask : torch.Tensor, optional
            A mask tensor to prevent attention to certain positions in the source sequence.
        src_key_padding_mask : torch.Tensor, optional
            A mask tensor to prevent attention to padding tokens in the source sequence.

        Returns
        -------
        torch.Tensor
            The encoded output of the source sequence. Shape: (batch_size, seq_len, d_model).
        """

        # Pass the source sequence through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return src


class DecoderBase(CoderBase):
    """
    Decoder class that builds on CoderBase for decoding sequences based on the encoder's memory.

    Parameters
    ----------
    d_model : int
        The dimension of the embedding space (output size of each layer).
    num_layers : int
        The number of layers in the decoder.
    layer_factory : nn.Module, optional
        Custom layer module. Defaults to `nn.TransformerDecoderLayer`.
    norm_layer : str or nn.Module, optional
        The normalization layer to apply. Can be a string or custom `nn.Module`. Default is 'LayerNorm'.
    dim_feedforward : int, optional
        The dimension of the feedforward network. Default is 2048.
    dropout : float, optional
        Dropout rate for regularization. Default is 0.1.
    num_heads : int, optional
        Number of attention heads in multi-head attention. Default is 8.
    factory_kwargs : dict, optional
        Additional arguments to pass to `layer_factory` when creating layers.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        layer_factory: Type[nn.Module] = nn.TransformerDecoderLayer,
        # norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        super(DecoderBase, self).__init__(
            d_model, num_layers, layer_factory, dim_feedforward, dropout, num_heads, **factory_kwargs
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        tgt : torch.Tensor
            The input tensor representing the target sequence. Shape: (batch_size, seq_len, d_model).
        memory : torch.Tensor
            The encoded memory output from the encoder. Shape: (batch_size, seq_len_src, d_model).
        tgt_mask : torch.Tensor, optional
            A mask tensor to prevent attention to certain positions in the target sequence.
        memory_mask : torch.Tensor, optional
            A mask tensor to prevent attention to certain positions in the memory sequence (from the encoder).
        tgt_key_padding_mask : torch.Tensor, optional
            A mask tensor to prevent attention to padding tokens in the target sequence.
        memory_key_padding_mask : torch.Tensor, optional
            A mask tensor to prevent attention to padding tokens in the memory sequence.

        Returns
        -------
        torch.Tensor
            The decoded output of the target sequence. Shape: (batch_size, seq_len_tgt, d_model).
        """

        # Pass the target sequence through each decoder layer
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return tgt
