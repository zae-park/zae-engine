from typing import OrderedDict, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from ..builds import BertBase, EncoderBase
from ...nn_night.layers import Additional

# implementation: # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
checkpoint_map = {
    "bert": "bert-base-uncased",
    "bert-small": "bert-base-uncased",
    "bert-large": "bert-base-uncased",
}


def __model_weight_mapper(src_weight: Union[OrderedDict | dict], dst_weight: Union[OrderedDict | dict]):
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
                k.replace("word_embeddings", "word")  # word
                .replace("position_embeddings", "position")  # position
                .replace("token_type_embeddings", "token_type")  # type
            )
            k = k.replace("embeddings", "encoder_embedding")
            k = k.replace("LayerNorm", "norm")  # norm
        elif k.startswith("encoder.layer"):
            k = k.replace(".layer.", ".layers.")
            # Save QKV weight & bias to buffer
            if "attention.self" in k:
                tkn = k.split(".")
                n_layer = tkn[2]
                para_name = "_".join(tkn[-2:])
                buff_dict[f"{n_layer}_{para_name}"] = v
                continue

            elif "attention.output" in k:
                k = (
                    k.replace("attention.output", "self_attn")
                    .replace("self_attn.LayerNorm", "norm1")
                    .replace("self_attn.dense", "self_attn.out_proj")
                )
            else:
                k = (
                    k.replace("intermediate.dense", "linear1")
                    .replace("output.dense", "linear2")
                    .replace("output.LayerNorm", "norm2")
                )

        elif k.startswith("pooler"):
            # Ignore Pooler layer
            continue
        else:
            raise NotImplementedError

        dst_weight[k] = v

    # Generate in_proj weight & bias using theirs of QKV in buffer
    total_layer = len(buff_dict) // 6
    for t in range(total_layer):
        qkv_w = [buff_dict[f"{t}_{n}_weight"] for n in ["query", "key", "value"]]
        qkv_b = [buff_dict[f"{t}_{n}_bias"] for n in ["query", "key", "value"]]
        dst_weight[f"encoder.layers.{t}.self_attn.in_proj_weight"] = torch.cat(qkv_w, dim=0)
        dst_weight[f"encoder.layers.{t}.self_attn.in_proj_bias"] = torch.cat(qkv_b, dim=0)
        # From : encoder.layer.0.attention.self.query.weight
        # To : encoder.layers.0.self_attn.in_proj_weight

    return dst_weight


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, dim_embedding):
        super(BertEmbedding, self).__init__()
        self.word = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.position = nn.Embedding(max_len, dim_embedding)
        self.token_type = nn.Embedding(2, dim_embedding)
        self.norm = nn.LayerNorm(dim_embedding)

    def forward(self, *input_args):
        w, p, t = input_args
        emb = self.word(w) + self.position(p) + self.token_type(t)
        return self.norm(emb)


def bert_base(pretrained=False) -> tuple:
    model_name = checkpoint_map["bert"]

    dim_model = 768
    dim_ff = 3072
    sep_token_id = 102
    src_vocab_size = 30522
    max_len = 512
    layer_factory = nn.TransformerEncoderLayer

    # Embedding = word + positional + type
    zae_emb = BertEmbedding(vocab_size=src_vocab_size, max_len=max_len, dim_embedding=dim_model)
    zae_enc = EncoderBase(d_model=dim_model, num_layers=12, layer_factory=layer_factory, dim_feedforward=dim_ff)
    zae_bert = BertBase(encoder_embedding=zae_emb, encoder=zae_enc, dim_hidden=dim_model, sep_token_id=sep_token_id)

    tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)

    if pretrained:
        pre_model = AutoModel.from_pretrained(model_name)
        new_weight = __model_weight_mapper(pre_model.state_dict(), zae_bert.state_dict())
        zae_bert.load_state_dict(new_weight, strict=True)

    return tokenizer, zae_bert
