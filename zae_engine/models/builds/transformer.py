import math

import torch
import torch.nn as nn


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
