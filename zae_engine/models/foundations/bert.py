from typing import OrderedDict, Union

from transformers import AutoModel, AutoTokenizer

from ..builds import transformer

checkpoint_map = {
    "bert": "bert-base-uncased",
}


def __model_weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):
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


def __tokenizer_weight_mapper(src_weight: [OrderedDict | dict], dst_weight: [OrderedDict | dict]):
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

    for k, v in src_weight.items():

        if k.startswith("layer"):
            k = k.replace(".bn", ".norm")
        elif k.startswith("fc"):
            pass
        else:
            k = "stem." + k
            k = k.replace("conv1", "0").replace("bn1", "1")

        dst_weight[k] = v

    return dst_weight


def bert_base(pretrained=False) -> transformer.UserIdModel:

    model = transformer.UserIdModel(num_classes=10)
    tokenizer = transformer.AuxModel(d_aux=1, d_model=32)  # TODO: Implement tokenizer
    if pretrained:
        model_name = checkpoint_map["bert"]

        pre_model = AutoModel.from_pretrained(model_name)
        new_weight = __model_weight_mapper(pre_model.parameters(), model.parameters())
        model.load_state_dict(new_weight, strict=True)

        pre_tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        if tokenizer:
            new_tokenizer = __tokenizer_weight_mapper(pre_tokenizer.parameters(), tokenizer.parameters())
            tokenizer.load_state_dict(new_tokenizer, strict=True)
        else:
            tokenizer = pre_tokenizer

    return model, tokenizer


#
# if __name__ == "__main__":

# # 모델과 토크나이저 로드
# model_name = "bert-base-uncased"  # 예: BERT 모델
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)  # DTrue
#
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
