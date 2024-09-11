import math
import copy
from typing import Callable, List, Type, Union, Tuple

import torch
import torch.nn as nn

from transformers.models.bert import BertModel
from transformers import AutoModel, AutoTokenizer


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
        layer_factory: Type[nn.Module] = nn.TransformerEncoderLayer,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        super(CoderBase, self).__init__()
        self.d_model = d_model

        # Set up normalization layer
        self.norm = self._get_norm_layer(norm_layer)

        # Create layers using the provided layer factory
        self.layers = nn.ModuleList(
            [
                layer_factory(
                    d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, **factory_kwargs
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization after all layers
        self.final_norm = self._get_norm_layer(norm_layer)

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
        layer_factory: Type[nn.Module] = nn.TransformerEncoderLayer,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        super(EncoderBase, self).__init__(
            d_model, num_layers, layer_factory, norm_layer, dim_feedforward, dropout, num_heads, **factory_kwargs
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
        # Apply initial normalization
        src = self.norm(src)

        # Pass the source sequence through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Apply final normalization after all layers
        output = self.final_norm(src)

        return output


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
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        super(DecoderBase, self).__init__(
            d_model, num_layers, layer_factory, norm_layer, dim_feedforward, dropout, num_heads, **factory_kwargs
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
        # Apply initial normalization
        tgt = self.norm(tgt)

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

        # Apply final normalization after all layers
        output = self.final_norm(tgt)

        return output


class Additional(nn.ModuleList):
    """
    Additional connection module.

    This module extends nn.ModuleList to implement an additional connection.
    Each input tensor is passed through its corresponding module,
    and the output tensors are summed. If the shapes of the output tensors
    do not match, an error is raised.

    Parameters
    ----------
    *args : nn.Module
        Sequence of PyTorch modules. Each module will be applied to
        a corresponding input tensor in the forward pass.

    Methods
    -------
    forward(*inputs)
        Applies each module to its corresponding input tensor and returns
        the sum of the output tensors. If the shapes of the output tensors
        do not match, an error is raised.
    """

    def __init__(self, *args):
        """
        Initialize the Additional module with a sequence of sub-modules.

        Parameters
        ----------
        *args : nn.Module
            Sequence of PyTorch modules to be applied to the input tensors.
        """
        super(Additional, self).__init__(args)

    def forward(self, *inputs):
        """
        Forward pass through the additional block.

        Applies each module to its corresponding input tensor and returns
        the sum of the output tensors. If the shapes of the output tensors
        do not match, an error is raised.

        Parameters
        ----------
        *inputs : torch.Tensor
            Sequence of input tensors. Each tensor is passed through its corresponding module.

        Returns
        -------
        torch.Tensor
            The sum of the output tensors of each module.

        Raises
        ------
        ValueError
            If the output tensors have mismatched shapes.
        """

        if len(inputs) != len(self):
            raise ValueError(f"Expected {len(self)} input tensors, but got {len(inputs)}.")

        # Apply each module to its corresponding input and store the outputs in a list
        outputs = [layer(inputs[i]) for i, layer in enumerate(self)]

        # Ensure that all output tensors have the same shape
        first_shape = outputs[0].shape
        for output in outputs[1:]:
            if output.shape != first_shape:
                raise ValueError(f"Shape mismatch: expected {first_shape}, but got {output.shape}")

        # Return the sum of the output tensors
        return sum(outputs)


def __weight_mapper(src_weight: [dict], dst_weight: [dict]):
    """
    Map source weights to destination model weights.

    Parameters
    ----------
    src_weight : OrderedDict or dict
        Source model weights.
    dst_weight : OrderedDict or dict
        Destination model weights.

    Returns
    -------
    OrderedDict or dict
        Updated destination model weights.
    """

    buff_dict = {}

    for k, v in src_weight.items():
        if k.startswith("embeddings"):
            k = (
                k.replace("word_embeddings", "0.0")  # word
                .replace("position_embeddings", "0.1")  # position
                .replace("token_type", "0.2")  # type
            )
            k = k.replace("embeddings", "encoder_embedding")
            k = k.replace("LayerNorm", "1")  # norm
        elif k.startswith("encoder.layer"):
            # Save QKV weight & bias to buffer
            if "attention.self" in k:
                tkn = k.split(".")
                n_layer = tkn[2]
                para_name = "_".join(tkn[-2:])
                buff_dict[f"{n_layer}_{para_name}"] = v

            elif "attention.output" in k:
                k = (
                    k.replace("layer.", "layers.")
                    .replace("attention.output", "self_attn")
                    .replace("dense", "out_proj")
                    .replace("LayerNorm", "norm")
                )

            elif "intermediate" in k:
                k = k.replace("intermediate.", "").replace("dense", "linear1")

            elif "output" in k:
                k = k.replace("output,", "").replace("dense", "linear2").replace("LayerNorm", "norm1")
                asd = "dense"
                asd = "LayerNorm"

            else:
                print(k)

        elif k.startswith("pooler"):
            pass
        else:
            print(k)

        dst_weight[k] = v

    # Generate in_proj weight & bias using theirs of QKV in buffer
    total_layer = len(buff_dict) // 6
    for t in total_layer:
        qkv_w = [buff_dict[f"{t}_{n}_weight"] for n in ["query", "key", "value"]]
        qkv_b = [buff_dict[f"{t}_{n}_bias"] for n in ["query", "key", "value"]]
        dst_weight[f"encoder.layers.{t}.self_attn.in_proj_weight"] = torch.cat(qkv_w, dim=0)
        dst_weight[f"encoder.layers.{t}.self_attn.in_proj_bias"] = torch.cat(qkv_b, dim=0)
        # From : encoder.layer.0.attention.self.query.weight
        # To : encoder.layers.0.self_attn.in_proj_weight

    return dst_weight


if __name__ == "__main__":
    # implementation: # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

    model_name = "bert-base-uncased"
    pre_tkn = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    pre_model = AutoModel.from_pretrained(model_name)

    bert_emb = pre_model.embeddings  # word_embeddings, position_embeddings, token_type_embeddings, LayerNorm, dropout
    bert_enc = pre_model.encoder  # BertEncoder
    bert_pool = pre_model.pooler  # BertPooler : Dense(768, 768) + Tanh()

    # Embedding = word + positional + type

    zae_emb = nn.Sequential(
        Additional(
            nn.Embedding(30522, 768, padding_idx=0),
            nn.Embedding(512, 768),
            nn.Embedding(2, 768),
        ),
        nn.LayerNorm(768),
    )

    encoder = EncoderBase(d_model=768, num_layers=12, layer_factory=nn.TransformerEncoderLayer, dim_feedforward=3072)
    decoder = nn.Identity()
    model = TransformerBase(encoder_embedding=zae_emb, encoder=encoder)

    enc = __weight_mapper(pre_model.state_dict(), model.state_dict())

    max_len = 16
    src_vocab_size = tgt_vocab_size = 1000
    src = torch.randint(0, src_vocab_size, (1, max_len))  # batch_size=32, seq_len=max_len
    tgt = torch.randint(0, tgt_vocab_size, (1, max_len))
    output = model(src, tgt)
    print(output.shape)  # Should return a tensor of shape (batch_size, seq_len, d_model)

    print()
