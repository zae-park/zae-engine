import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeAwareTransformer(nn.Module):
    """
    Time-aware Transformer model for encoding event sequences with time information.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    nhead : int
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

    def __init__(self, d_model, nhead, num_layers, max_seq_len=512):
        super(TimeAwareTransformer, self).__init__()
        self.embedding = nn.Linear(128, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        self.time_encoder = nn.Embedding(max_seq_len, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

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
        seq_len, batch_size, _ = event_vecs.size()
        pos_encoding = self.pos_encoder(torch.arange(seq_len, device=event_vecs.device)).unsqueeze(1)
        time_encoding = self.time_encoder(time_vecs)

        event_vecs = self.embedding(event_vecs) + pos_encoding + time_encoding
        transformer_out = self.transformer_encoder(event_vecs, src_key_padding_mask=mask)

        output = transformer_out[-1, :, :]
        return output
