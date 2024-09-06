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
    forward(src, tgt=None, src_mask=None, tgt_mask=None)
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
        self.decoder_embedding = decoder_embedding if decoder_embedding is not None else encoder_embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
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

        Returns
        -------
        torch.Tensor
            If a decoder is provided, returns the output of the decoder. Otherwise, returns the output of the encoder.
        """
        # Apply embeddings to source and target sequences
        src_embed = self.encoder_embedding(src)

        # If a decoder exists, apply decoder embedding and pass through the decoder
        if self.decoder is not None and tgt is not None:
            tgt_embed = self.decoder_embedding(tgt)
            encoded = self.encoder(src_embed, src_mask)
            out = self.decoder(tgt_embed, encoded, src_mask, tgt_mask)
            return out
        else:
            # If no decoder, only pass through the encoder
            return self.encoder(src_embed, src_mask)


class EncoderBase(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        layer_factory: nn.Module = nn.TransformerEncoderLayer,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_heads: int = 8,
        **factory_kwargs,
    ):
        """
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
        factory_kwargs : dict
            Additional arguments to pass to `layer_factory` when creating layers.
        """
        super(EncoderBase, self).__init__()

        # Set up normalization layer
        self.norm = self._get_norm_layer(norm_layer, d_model)

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
        self.final_norm = self._get_norm_layer(norm_layer, d_model)

    def _get_norm_layer(self, norm_type, d_model):
        """
        Returns the appropriate normalization layer based on user input.
        """
        if isinstance(norm_type, nn.Module):
            return norm_type
        elif norm_type == "LayerNorm":
            return nn.LayerNorm(d_model)
        elif norm_type == "BatchNorm1d":
            return nn.BatchNorm1d(d_model)
        elif norm_type == "InstanceNorm1d":
            return nn.InstanceNorm1d(d_model)
        elif norm_type == "GroupNorm":
            return nn.GroupNorm(8, d_model)
        else:
            raise ValueError(f"Unsupported norm layer type: {norm_type}")

    def forward(self, src, src_mask=None):
        """
        Forward pass through the encoder.
        """
        # Apply initial normalization
        src = self.norm(src)

        # Pass the source sequence through each encoder layer
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_mask)

        # Apply final normalization after all layers
        output = self.final_norm(src)

        return output


class DecoderBase(EncoderBase):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass through the decoder.
        """
        # Apply initial normalization
        tgt = self.norm(tgt)

        # Pass the target sequence through each decoder layer
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Apply final normalization after all layers
        output = self.final_norm(tgt)

        return output


class EncoderBaseLegacy(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        vocab_size: int,
        max_length: int,
        d_model: int,
        d_head: int,
        n_head: int,
        n_layers: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.n_layers = n_layers

        self.emb_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self.activation = kwargs.pop("activation", "gelu")
        self.norm_first = kwargs.pop("norm_first", False)

        if not isinstance(layer, nn.Module):
            layer = nn.TransformerEncoderLayer(d_model, n_head, activation=self.activation, norm_first=self.norm_first)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        # check args: is_causal, src_key_padding_mask

    def forward(self, src, src_mask):
        """
        :param src: (batch_size) embedded.
        :param src_mask: (batch_size) mask.
        Parameters
        ----------
        src
        src_mask

        Returns
        -------

        """
        return self.transformer_encoder(src, mask=src_mask)


def timestamp_encoding(timestamps, d_model):
    """
    Encode timestamp to periodic

    Args:
        timestamps (torch.Tensor): Sequence of timestamp with (batch_size, seq_len) shape.
        d_model (int): dimension of model.

    Returns:
        torch.Tensor: Timestamp encoding tensor with (batch_size, seq_len, d_model) shape.
    """
    batch_size, seq_len = timestamps.size()
    pe = torch.zeros(batch_size, seq_len, d_model, device=timestamps.device)
    position = timestamps.unsqueeze(-1)  # [Batch, sequence_length, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float().to(timestamps.device) * -(math.log(10000.0) / d_model))

    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)

    return pe


class TimeAwareTransformer(nn.Module):
    """
    Time-aware Transformer model for encoding event sequences with time information.

    Parameters
    ----------
    d_head : int
        Dimension of the head.
    d_model : int
        Dimension of the model.
    n_head : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    max_seq_len : int, optional
        Maximum sequence length (default is 512).

    Methods
    -------
    forward(event_vecs, time_vecs, mask=None):
        Forward pass through the model.
    """

    def __init__(self, d_head: int, d_model: int, n_head: int, num_layers: int, max_seq_len: int = 512, **kwargs):
        super(TimeAwareTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(d_head, d_model)  # 이벤트 벡터를 d_model 차원으로 변환
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, activation="gelu", norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, event_vecs, time_vecs, mask=None):
        """
        Forward pass through the TimeAwareTransformer.

        Parameters
        ----------
        event_vecs : torch.Tensor
            Input event vectors of shape (seq_len, batch_size, 128).
        time_vecs : torch.Tensor
            Input time vectors of shape (seq_len, batch_size).
        mask : torch.Tensor, optional
            Mask for the input data (default is None).

        Returns
        -------
        torch.Tensor
            Output features of the model of shape (batch_size, d_model).
        """
        batch_size, max_seq_len, _ = event_vecs.size()

        # add positional encoding
        pos_indices = torch.arange(max_seq_len, device=event_vecs.device)
        pos_encoding = self.pos_encoder(pos_indices).unsqueeze(0).repeat(batch_size, 1, 1)

        # add timestamp encoding
        time_encoding = timestamp_encoding(time_vecs, self.d_model)

        # apply encodings to event vectors
        event_vecs = self.embedding(event_vecs) + pos_encoding + time_encoding
        event_vecs = event_vecs.permute(1, 0, 2)  # (max_seq_len, batch_size, d_model)

        transformer_out = self.transformer_encoder(event_vecs, src_key_padding_mask=mask)
        output = transformer_out[-1, :, :]  # use last sequence vector
        return output


class AuxModel(nn.Module):
    def __init__(self, d_aux, d_model):
        super(AuxModel, self).__init__()
        self.aux_fc = nn.Linear(d_aux, d_model)  # aux_vector를 d_model 차원으로 변환
        self.aux_act = nn.ReLU()

    def forward(self, aux_vector):
        return self.aux_act(self.aux_fc(aux_vector))


class UserIdModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(UserIdModel, self).__init__()
        self.transformer = TimeAwareTransformer(**kwargs["model"].event_body._asdict())
        self.aux_body = AuxModel(**kwargs["model"].aux_body._asdict())

        dim_event = kwargs["model"].event_body.d_model
        dim_aux = kwargs["model"].aux_body.d_model
        self.neck = nn.Linear(dim_event + dim_aux, dim_event)  # combined_feature를 d_model 차원으로 변환
        self.neck_act = nn.ReLU()

        self.num_classes = num_classes

    def forward(self, event_vecs, time_vecs, aux, mask=None):
        features = self.transformer(event_vecs, time_vecs, mask)
        aux_feature = self.aux_body(aux)
        combined_features = torch.cat((features, aux_feature), dim=1)
        out = self.neck_act(self.neck(combined_features))
        return out

    # def expand_classes(self, new_num_classes):
    #     old_weight = self.arcface.weight.data
    #     self.arcface = ArcFaceLoss(self.arcface.weight.shape[1], new_num_classes)
    #     self.arcface.weight.data[: self.num_classes] = old_weight
    #     self.num_classes = new_num_classes


if __name__ == "__main__":

    # Define hyperparameters
    d_model = 768
    num_layers = 6
    max_len = 512
    src_vocab_size = 10000
    tgt_vocab_size = 10000

    src_emb = nn.Embedding(src_vocab_size, d_model)
    tgt_emb = nn.Embedding(src_vocab_size, d_model)
    encoder = EncoderBase(
        d_model=d_model, num_layers=num_layers, layer_factory=nn.TransformerEncoderLayer, dim_feedforward=3072
    )
    decoder = DecoderBase(
        d_model=d_model, num_layers=num_layers, layer_factory=nn.TransformerDecoderLayer, dim_feedforward=3072
    )
    model = TransformerBase(encoder_embedding=src_emb, decoder_embedding=tgt_emb, encoder=encoder, decoder=decoder)

    # Sample input data
    src = torch.randint(0, src_vocab_size, (32, max_len))  # batch_size=32, seq_len=max_len
    tgt = torch.randint(0, tgt_vocab_size, (32, max_len))

    # Run a forward pass through the Transformer
    output = model(src, tgt)
    print(output.shape)  # Should return a tensor of shape (batch_size, seq_len, d_model)

    model_name = "bert-base-uncased"
    pre_tkn = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    pre_model = AutoModel.from_pretrained(model_name)

    bert_emb = pre_model.embeddings  # word_embeddings, position_embeddings, token_type_embeddings, LayerNorm, dropout
    bert_enc = pre_model.encoder  # BertEncoder
    bert_pool = pre_model.pooler  # BertPooler : Dense(768, 768) + Tanh()

    # Embedding = word + positional + type
    zae_emb = nn.ModuleList([nn.Embedding(30522, 768, padding_idx=0), nn.Embedding(512, 768), nn.Embedding(2, 768)])
    zae_embedding = nn.Sequential(nn.Embedding(30522 + 512 + 2, 768, padding_idx=0), nn.LayerNorm(768), nn.Dropout(0.1))

    encoder = EncoderBase(d_model=768, num_layers=12, layer_factory=nn.TransformerEncoderLayer, dim_feedforward=3072)
    decoder = nn.Identity()
    model = TransformerBase(encoder_embedding=zae_embedding, encoder=encoder)
    output = model(src, tgt)
    print(output.shape)  # Should return a tensor of shape (batch_size, seq_len, d_model)

    print()
