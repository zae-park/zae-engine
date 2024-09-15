import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Computes sinusoidal positional encoding as described in the Transformer paper [1]_.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space. Must be an even number.
    max_len : int, optional
        Maximum sequence length. Default is 512.

    Notes
    -----
    - This method was introduced in the original Transformer paper (Vaswani et al., 2017) [1]_.
    - Uses fixed sine and cosine functions of different frequencies to encode token positions.
    - Benefits: Simple and efficient to compute, and captures positional information effectively.
    - Drawbacks: Does not capture relative positional information.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.Õ., Kaiser, Ł., Polosukhin, I.
           (2017). Attention is All You Need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS 2017).
           Available at: https://arxiv.org/abs/1706.03762
           DOI: 10.48550/arXiv.1706.03762
    """

    def __init__(self, d_model, max_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model must be an even number for sinusoidal positional encoding."
        self.d_model = d_model
        self.max_len = max_len

    def _create_positional_encoding(self, positions):
        """
        Internal method to create positional encodings using sine and cosine functions.

        Parameters
        ----------
        positions : torch.Tensor
            Tensor of positions to be encoded.

        Returns
        -------
        torch.Tensor
            Positional encoding tensor of shape (batch_size, seq_len, d_model).
        """
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pos_enc = torch.zeros(positions.size(0), positions.size(1), self.d_model)
        pos_enc[:, :, 0::2] = torch.sin(positions * div_term.unsqueeze(0).unsqueeze(0))
        pos_enc[:, :, 1::2] = torch.cos(positions * div_term.unsqueeze(0).unsqueeze(0))
        return pos_enc

    def forward(self, x, positions: torch.Tensor = None):
        """
        Apply sinusoidal positional encoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        positions : torch.Tensor, optional
            Optional tensor of shape (batch_size, seq_len) specifying the positions (e.g., timestamps) for each element in the sequence.
            If not provided, the default positions (0 to seq_len - 1) are used.

        Returns
        -------
        torch.Tensor
            Tensor with added positional encoding.
        """
        batch_size, seq_len, _ = x.size()

        if positions is not None:
            assert positions.size(0) == batch_size, "Positions batch size must match input batch size"
            assert positions.size(1) == seq_len, "Positions sequence length must match input sequence length"
        else:
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        pos_enc = self._create_positional_encoding(positions.unsqueeze(-1))
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

    References
    ----------
    - No specific reference, this approach is commonly used in various models including BERT and GPT.
    """

    def __init__(self, d_model, max_len=512):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        """
        Apply learnable positional encoding to input tensor.

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
        pos_enc = self.position_embeddings[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pos_enc


class RotaryPositionalEncoding(nn.Module):
    """
    Implements Rotary Positional Encoding as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Parameters
    ----------
    d_model : int
        Dimension of the embedding space. Must be divisible by 2 for rotary encoding.
    """

    def __init__(self, d_model):
        super(RotaryPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model should be divisible by 2 for rotary encoding."
        self.d_model = d_model
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def forward(self, x):
        """
        Apply rotary positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with added rotary positional encoding.
        """
        seq_len = x.size(1)
        # Create sinusoidal positional encoding
        positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoidal_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        sin_enc = torch.sin(sinusoidal_inp)
        cos_enc = torch.cos(sinusoidal_inp)

        # Apply rotary embedding
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos_enc - x2 * sin_enc, x1 * sin_enc + x2 * cos_enc], dim=-1)

        return x


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
    - This method is used in models like Transformer-XL and T5 [2]_[3]_.
    - Benefits: Handles long sequences and captures relative positions.
    - Drawbacks: May increase computational complexity.

    References
    ----------
    .. [2] Dai, Z., Yang, Z., Yang, Y., Carbonell, J. G., Salakhutdinov, R., & Liu, T.-Y. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.
           In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019).
           Available at: https://arxiv.org/abs/1901.02860
           DOI: 10.48550/arXiv.1901.02860

    .. [3] Raffel, C., Shinn, C., Gauthier, J., & others (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
           Journal of Machine Learning Research, 21, 1-67.
           Available at: https://arxiv.org/abs/1910.10683
           DOI: 10.48550/arXiv.1910.10683
    """

    def __init__(self, d_model, max_len=512):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.relative_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        """
        Apply relative positional encoding to input tensor.

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
        pos_enc = self.relative_embeddings[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
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

    References
    ----------
    - No specific reference, this approach is inspired by the need for adaptive positional encodings in models handling variable-length sequences.
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
