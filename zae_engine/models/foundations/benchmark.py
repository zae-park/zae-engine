import torch
import torch.nn as nn

from ..builds import transformer as trx
from ...nn_night.layers import SinusoidalPositionalEncoding
from ...loss import angular


# class UserIdModel(nn.Module):
#     """
#     User identification model using a Transformer encoder and ArcFace loss.
#
#     Parameters
#     ----------
#     d_model : int
#         Dimension of the model.
#     n_head : int
#         Number of attention heads.
#     num_layers : int
#         Number of transformer encoder layers.
#     num_classes : int
#         Number of user classes.
#
#     Methods
#     -------
#     forward(event_vecs, time_vecs, labels, mask=None):
#         Forward pass through the model.
#     expand_classes(new_num_classes):
#         Expand the number of classes dynamically.
#     """
#
#     def __init__(self, d_model, n_head, num_layers, num_classes):
#         super(UserIdModel, self).__init__()
#         self.transformer = trx.EncoderBase(d_model=d_model, num_layers=num_layers, num_heads=n_head)
#         self.arcface = angular.ArcFaceLoss(d_model, num_classes)
#         self.num_classes = num_classes
#
#     def forward(self, event_vecs, time_vecs, labels, mask=None):
#         """
#         Forward pass through the UserIdentificationModel.
#
#         Parameters
#         ----------
#         event_vecs : torch.Tensor
#             Input event vectors of shape (seq_len, batch_size, 128).
#         time_vecs : torch.Tensor
#             Input time vectors of shape (seq_len, batch_size).
#         labels : torch.Tensor
#             Ground truth labels of shape (batch_size).
#         mask : torch.Tensor, optional
#             Mask for the input data (default is None).
#
#         Returns
#         -------
#         tuple of torch.Tensor
#             Logits of shape (batch_size, num_classes) and features of shape (batch_size, d_model).
#         """
#         features = self.transformer(event_vecs, time_vecs, mask)
#         logits = self.arcface(features, labels)
#         return logits, features
#
#     def expand_classes(self, new_num_classes):
#         """
#         Expand the number of classes dynamically.
#
#         Parameters
#         ----------
#         new_num_classes : int
#             New number of classes.
#         """
#         old_weight = self.arcface.weight.data
#         self.arcface = angular.ArcFaceLoss(self.arcface.weight.shape[1], new_num_classes)
#         self.arcface.weight.data[: self.num_classes] = old_weight
#         self.num_classes = new_num_classes


class TimeSeriesTransformer(nn.Module):
    """
    A Transformer model for time series data using BertBase and SinusoidalPositionalEncoding.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    d_model : int
        The dimension of the embedding space.
    max_len : int
        The maximum sequence length.
    num_layers : int
        The number of layers in the Transformer encoder.
    num_heads : int
        The number of attention heads in the Transformer encoder.
    dim_feedforward : int
        The dimension of the feedforward network in the Transformer encoder.
    dropout : float
        The dropout rate for regularization.
    """

    def __init__(self, vocab_size, d_model, max_len, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding layer
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        # Transformer encoder
        self.encoder = trx.EncoderBase(
            d_model=d_model,
            num_layers=num_layers,
            layer_factory=nn.TransformerEncoderLayer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_heads=num_heads,
        )

        # BertBase for the transformer
        self.transformer = trx.BertBase(
            encoder_embedding=self._embedding_with_position, encoder=self.encoder, dim_hidden=d_model
        )

    def _embedding_with_position(self, input_ids, positions, token_type_ids=None):
        """
        Embeds the input tokens and adds positional encoding.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of input token IDs with shape (batch_size, seq_len).
        positions : torch.Tensor
            Tensor of positions (timestamps) with shape (batch_size, seq_len).
        token_type_ids : torch.Tensor, optional
            Tensor of token type IDs. Not used in this implementation.

        Returns
        -------
        torch.Tensor
            The embedded input with positional encoding added.
        """
        # Get word embeddings
        word_embeds = self.word_embedding(input_ids)

        # Add positional encoding
        word_embeds_with_pos = self.positional_encoding(word_embeds, positions)
        return word_embeds_with_pos

    def forward(self, input_ids, positions, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for the time series Transformer model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of input token IDs with shape (batch_size, seq_len).
        positions : torch.Tensor
            Tensor of positions (timestamps) with shape (batch_size, seq_len).
        src_mask : torch.Tensor, optional
            Source mask for masking certain positions in the input.
        src_key_padding_mask : torch.Tensor, optional
            Mask for padding tokens in the input sequence.

        Returns
        -------
        torch.Tensor
            Output from the encoder, shape (batch_size, seq_len, d_model).
        """
        return self.transformer(input_sequence=input_ids, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
