from typing import OrderedDict, Union

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..builds import transformer as trns

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

    # pretrained weight
    bert_model = "base"

    embedding = "BertEmbeddings"
    embdding_ = {
        "word_embeddings": nn.Embedding(30522, 768, padding_idx=0),
        "position_embeddings": nn.Embedding(512, 768, padding_idx=0),
        "token_type_embeddings": nn.Embedding(2, 768, padding_idx=0),
        "LayerNorm": nn.LayerNorm(768),
        "dropout": nn.Dropout(0.1),
    }

    encoder = "BertEncoder"
    encoder_ = {
        "layer": [
            {
                "0-11": {
                    "12 x BertLayer": {
                        "attention": {
                            "BertAttention": {
                                "self": ["BertSelfAttention", ["query", "key", "value", "768->768"], "dropout(0.1)"],
                                "output": ["BertSelfOutput", ["dense", "LayerNorm", "dropout"]],
                            }
                        },
                        "intermediate": {
                            "BertIntermediate": {"dense": nn.Linear(768, 3072), "intermediate_act_fn": nn.Gelu()}
                        },
                        "output": {
                            "BertOutput": {
                                "dense": nn.Linear(3072, 768),
                                "LayerNorm": nn.LayerNorm(768),
                                "dropout": nn.Dropout(0.1),
                            }
                        },
                    }
                }
            }
        ]
    }

    pooler = "BertPooler"
    pooler_ = {"dense": nn.Linear(768, 768), "activation": nn.Tanh()}
    #

    for k, v in src_weight.items():

        if k.startswith("layer"):
            k = (
                k.replace("layer1", "body.0")
                .replace("layer2", "body.1")
                .replace("layer3", "body.2")
                .replace("layer4", "body.3")
            )
            k = k.replace(".bn", ".norm")
        elif k.startswith("fc"):
            pass
        else:
            k = "stem." + k
            k = k.replace("conv1", "0").replace("bn1", "1")

        dst_weight[k] = v

    return dst_weight


# def bert_base(pretrained=False, tokenizer_name: Union[str, None] = None) -> tuple:
#     model_name = checkpoint_map["bert"]
#
#     # zae_model = transformer.EncoderBase(layer=nn.TransformerEncoder)
#
#     if tokenizer_name is None:
#         tokenizer_name = model_name
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)
#
#     model = trns.TransformerBase(nn.Embedding(1, 2), encoder=nn.Identity())
#     if pretrained:
#         pre_model = AutoModel.from_pretrained(model_name)
#         new_weight = __model_weight_mapper(pre_model.parameters(), model.parameters())
#         model.load_state_dict(new_weight, strict=True)
#
#     return tokenizer, model


if __name__ == "__main__":

    model_name = "bert-base-uncased"
    pre_tkn = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    pre_model = AutoModel.from_pretrained(model_name)

    bert_emb = pre_model.embeddings  # word_embeddings, position_embeddings, token_type_embeddings, LayerNorm, dropout
    bert_enc = pre_model.encoder  # BertEncoder
    bert_pool = pre_model.pooler  # BertPooler : Dense(768, 768) + Tanh()

    # Embedding = word + positional + type
    zae_emb = nn.ModuleList([nn.Embedding(30522, 768, padding_idx=0), nn.Embedding(512, 768), nn.Embedding(2, 768)])
    zae_embedding = nn.Sequential(nn.Embedding(30522 + 512 + 2, 768, padding_idx=0), nn.LayerNorm(768), nn.Dropout(0.1))

    encoder = trns.EncoderBase(
        d_model=768, num_layers=12, layer_factory=nn.TransformerEncoderLayer, dim_feedforward=3072
    )
    decoder = nn.Identity()
    model = trns.TransformerBase(encoder_embedding=zae_embedding, encoder=encoder)
    import torch

    max_len = 16
    src_vocab_size = tgt_vocab_size = 1000
    src = torch.randint(0, src_vocab_size, (1, max_len))  # batch_size=32, seq_len=max_len
    tgt = torch.randint(0, tgt_vocab_size, (1, max_len))
    output = model(src, tgt)
    print(output.shape)  # Should return a tensor of shape (batch_size, seq_len, d_model)

    print()
