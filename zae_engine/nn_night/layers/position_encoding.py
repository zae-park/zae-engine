import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Computes sinusoidal positional encoding as described in the Transformer paper.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space.
    max_len : int, optional
        Maximum sequence length. Default is 512.

    Notes
    -----
    - This method was introduced in the original Transformer paper (Vaswani et al., 2017).
    - Uses fixed sine and cosine functions of different frequencies.
    - Benefits: Simple and efficient to compute.
    - Drawbacks: Does not capture relative positional information.
    """

    def __init__(self, d_model, max_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embeddings = self._create_positional_encoding()

    def _create_positional_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pos_enc = torch.zeros(self.max_len, self.d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        """
        Apply sinusoidal positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with added positional encoding.
        """
        seq_len = x.size(1)
        pos_enc = self.position_embeddings[:seq_len]
        pos_enc = pos_enc.unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pos_enc


class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encoding where positional embeddings are learned during training.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space.
    max_len : int, optional
        Maximum sequence length. Default is 512.

    Notes
    -----
    - This method allows the model to learn optimal positional encodings during training.
    - Benefits: Can adapt the positional encoding to the specific task.
    - Drawbacks: Requires additional parameters and training time.
    """

    def __init__(self, d_model, max_len=512):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.position_embeddings[:seq_len]
        return x + pos_enc


class RotaryPositionalEncoding(nn.Module):
    """
    Implements rotary positional encoding that uses rotation to encode relative positions.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space.

    Notes
    -----
    - This method was introduced in the RoFormer model.
    - Benefits: Captures relative positional information effectively.
    - Drawbacks: Computationally more complex compared to sinusoidal encoding.
    """

    def __init__(self, d_model):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        angle = torch.arange(self.d_model // 2).float() / (self.d_model // 2)
        angle = 1 / (10000**angle)
        angle = angle.unsqueeze(0).expand(seq_len, -1)
        angle = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

        x_rot = x * angle
        return x_rot


class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encoding that captures relative distances between tokens.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space.
    max_len : int, optional
        Maximum sequence length. Default is 512.

    Notes
    -----
    - This method is used in models like Transformer-XL and T5.
    - Benefits: Handles long sequences and captures relative positions.
    - Drawbacks: May increase computational complexity.
    """

    def __init__(self, d_model, max_len=512):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.relative_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.relative_embeddings[:seq_len]
        return x + pos_enc


class AdaptivePositionalEncoding(nn.Module):
    """
    Implements adaptive positional encoding that adjusts position encoding based on input sequence length.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space.
    max_len : int, optional
        Maximum sequence length. Default is 512.

    Notes
    -----
    - This method adjusts the position encoding dynamically based on sequence length.
    - Benefits: Flexible for handling sequences of varying lengths.
    - Drawbacks: Requires additional handling for sequences with different lengths.
    """

    def __init__(self, d_model, max_len=512):
        super(AdaptivePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x, **kwargs):
        """
        Apply adaptive positional encoding to input tensor based on sequence length.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        **kwargs : dict
            Additional keyword arguments, such as 'seq_lengths' for handling variable-length sequences.

        Returns
        -------
        torch.Tensor
            Tensor with added positional encoding.
        """
        seq_lengths = kwargs.get("seq_lengths")
        if seq_lengths is None:
            seq_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)

        batch_size, seq_len, _ = x.size()
        output = torch.zeros_like(x)
        for i in range(batch_size):
            length = seq_lengths[i]
            pos_enc = self.position_embeddings[:length]
            pos_enc = pos_enc.unsqueeze(0).expand(1, -1, -1)
            output[i, :length, :] = x[i, :length, :] + pos_enc

        return output
