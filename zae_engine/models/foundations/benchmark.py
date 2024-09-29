import torch
import torch.nn as nn

from ..builds import transformer as trx
from ...nn_night.layers import SinusoidalPositionalEncoding


class TimeSeriesBert(nn.Module):
    """
    Encoder-only Transformer model based on BertBase with integrated embedding and positional encoding.

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
    dropout : float, optional
        The dropout rate for regularization. Default is 0.1.
    dim_pool : int, optional
        The hidden dimension for the pooler. If provided, a pooler is applied to the [CLS] token.
    sep_token_id : int, optional
        The token ID for [SEP]. Default is 102.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        dim_pool: int = None,
        sep_token_id: int = 102,
        **factory_kwargs,
    ):
        super(TimeSeriesBert, self).__init__()

        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        # Transformer encoder
        self.encoder = trx.EncoderBase(
            d_model=d_model,
            num_layers=num_layers,
            layer_factory=nn.TransformerEncoderLayer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_heads=num_heads,
            **factory_kwargs,
        )

        # Optional pooler
        self.dim_pool = dim_pool
        self.sep_token_id = sep_token_id
        if self.dim_pool:
            self.pool_dense = nn.Linear(self.dim_pool, self.dim_pool)
            self.pool_activation = nn.Tanh()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder-only Transformer model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of input token IDs with shape (batch_size, seq_len).
        positions : torch.Tensor, optional
            Tensor of positions (timestamps) with shape (batch_size, seq_len).
        src_mask : torch.Tensor, optional
            Source mask for masking certain positions in the input. Shape: (seq_len, seq_len).
        src_key_padding_mask : torch.Tensor, optional
            Mask for padding tokens in the input sequence. Shape: (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Output from the encoder or pooled output if dim_pool is set.
        """
        # Get word embeddings
        word_embeds = self.word_embedding(input_ids)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        word_embeds_with_pos = self.positional_encoding(word_embeds, positions)  # [batch_size, seq_len, d_model]

        # Pass through encoder
        encoded_output = self.encoder(
            word_embeds_with_pos, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, d_model]

        # Apply pooler if specified
        if self.dim_pool:
            cls_tkn = encoded_output[:, 0, :]  # [CLS] token [batch_size, d_model]
            pooled_output = self.pool_activation(self.pool_dense(cls_tkn))  # [batch_size, dim_pool]
            return pooled_output

        return encoded_output
