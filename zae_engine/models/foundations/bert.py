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


def bert_base(pretrained=False, tokenizer_name: Union[str, None] = None) -> trns.UserIdModel:
    model_name = checkpoint_map["bert"]

    # zae_model = transformer.EncoderBase(layer=nn.TransformerEncoder)

    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)

    model = transformer.UserIdModel(num_classes=10)
    if pretrained:
        pre_model = AutoModel.from_pretrained(model_name)
        new_weight = __model_weight_mapper(pre_model.parameters(), model.parameters())
        model.load_state_dict(new_weight, strict=True)

    return tokenizer, model



if __name__ == "__main__":

    pre_tkn, pre_model = bert_base()

    # Define hyperparameters
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    dropout = 0.1
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    max_len = 100
    num_layers = 6


    # extract embedding layer
    src_emb = nn.Embedding(src_vocab_size, 768)
    tgt_emb = nn.Embedding(src_vocab_size, 768)
    

    # Encoder and Decoder using EncoderBase and DecoderBase
    encoder = trns.EncoderBase(
        d_model=d_model,
        num_layers=num_layers,
        transformer_layer=nn.TransformerEncoderLayer
    )
    decoder = trns.DecoderBase(
        d_model=d_model,
        num_layers=num_layers,
        transformer_layer=nn.TransformerDecoderLayer
    )


    # Build the Transformer
    model = trns.TransformerBase(
    encoder_embedding=src_emb,
    decoder_embedding=tgt_emb,
    encoder=encoder,
    decoder=decoder
)

    # Sample input data
    src = torch.randint(0, src_vocab_size, (32, max_len))  # batch_size=32, seq_len=max_len
    tgt = torch.randint(0, tgt_vocab_size, (32, max_len))

    # Run a forward pass through the Transformer
    output = model(src, tgt)
    print(output.shape)  # Should return a tensor of shape (batch_size, seq_len, d_model)


    # Example of a simple test with PyTorch's Transformer layers
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)

    # Define custom positional encoding and embeddings
    src_embedding = nn.Sequential(nn.Embedding(10000, 512), SinusoidalPositionalEncoding(d_model=512))
    tgt_embedding = nn.Sequential(nn.Embedding(10000, 512), SinusoidalPositionalEncoding(d_model=512))

    # Define the encoder and decoder with flexibility for different transformer layers
    encoder = EncoderBase(d_model=512, num_layers=6, transformer_layer=encoder_layer)
    decoder = DecoderBase(d_model=512, num_layers=6, transformer_layer=decoder_layer)

    # Define the Transformer model
    model = TransformerBase(encoder_embedding=src_embedding, decoder_embedding=tgt_embedding, encoder=encoder, decoder=decoder)

    # Run a sample input
    src = torch.randint(0, 10000, (32, 100))  # (batch_size, seq_len)
    tgt = torch.randint(0, 10000, (32, 100))
    output = model(src, tgt)
    print(output.shape)



# # 모델과 토크나이저 로드
# model_name = "bert-base-uncased"  # 예: BERT 모델
# model = AutoModel.from_pretrained(model_name)


# # 예제 텍스트를 토큰화하고 모델에 입력
# text = "Hello, world!"
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model(**inputs)
#
# print(outputs)

# import platform
#
# pf = platform.platform().lower()
#
# if "windows" in pf:
#     # https://ollama.com/download/OllamaSetup.exe
#     print("Windows")
# elif "darwin" in pf:
#     # https://ollama.com/download/Ollama-darwin.zip
#     print("macOS")
# elif "linux" in pf:
#     # sh$ curl -fsSL https://ollama.com/install.sh | sh
#     print("Linux")
# else:
#     raise Exception

#     from transformers.models import llama as llm
#     from transformers.models.llama import convert_llama_weights_to_hf as cvt
#     # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/model_checkpointing/checkpoint_handler.py
#     cvt.write_model(
#         model_path="./output_dir",
#         input_base_path="./input_dir",
#         model_size=args.model_size,
#         safe_serialization=args.safe_serialization,
#         llama_version=args.llama_version,
#         vocab_size=vocab_size,
#         num_shards=args.num_shards,
#         instruct=args.instruct,
#     )
