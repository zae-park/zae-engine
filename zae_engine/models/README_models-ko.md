# Models 서브패키지 개요

**zae-engine** 라이브러리의 **Models 서브패키지**는 다양한 딥러닝 모델을 구축하고 확장하기 위한 다양한 기본 빌드와 미리 정의된 모델들을 제공합니다. 이 서브패키지는 CNN, Transformer, AutoEncoder, UNet, ResNet, BERT 등의 모델을 포함하여, 다양한 딥러닝 아키텍처를 쉽게 정의하고 활용할 수 있는 도구들을 제공합니다.

## 주요 구성 요소 및 모델들

### Builds 하위 모듈

`builds` 모듈은 딥러닝 모델의 구성 요소를 만드는 기본적인 빌드 클래스를 제공합니다. 예를 들어, **CNNBase**, **AutoEncoder**, **TransformerBase** 등의 클래스가 포함되어 있으며, 이러한 클래스들은 네트워크의 기반을 구성하는 데 사용할 수 있습니다.

- **CNNBase**: CNN 아키텍처의 기본 구조를 제공합니다. 이 클래스는 입력 이미지를 받아 여러 레이어를 통과시키며 특징을 추출하는 구조를 정의합니다.
- **AutoEncoder**: 기본적인 AutoEncoder 아키텍처를 제공하며, U-Net 스타일의 스킵 연결을 통해 이미지 재구성을 할 수 있는 기능도 포함하고 있습니다.
- **TransformerBase**: Transformer 아키텍처를 위한 기본 클래스를 제공하며, 인코더와 디코더 구조를 지원합니다.

### Foundations 하위 모듈

`foundations` 모듈은 잘 알려진 사전 훈련된 모델이나 특정 작업을 위한 모델을 구현합니다. 예를 들어, **BERT**, **ResNet**, **UNet** 등의 모델을 포함하며, 사전 훈련된 가중치를 로드하여 빠르게 사용할 수 있습니다.

- **BertBase**: BERT 아키텍처의 기본 구현입니다. 사전 훈련된 BERT 모델의 가중치를 로드하여 텍스트 임베딩 및 NLP 작업에 사용할 수 있습니다.
- **ResNet**: ResNet-18, ResNet-34, ResNet-50 등의 다양한 버전의 ResNet을 제공합니다. 이미지 인식 작업에 널리 사용됩니다.
- **UNet**: U-Net 아키텍처로, 주로 바이오메디컬 이미지 분할에 사용됩니다. 사전 훈련된 가중치를 사용하여 빠르게 모델을 구축할 수 있습니다.

### Converter 하위 모듈

`converter` 모듈은 모델의 특정 레이어를 변환하거나 확장하는 기능을 제공합니다. 예를 들어, 차원을 변환하거나 새로운 모듈을 주입하여 기존 네트워크를 수정할 수 있습니다.

- **DimConverter**: 모델 내 특정 레이어의 차원을 변환할 수 있습니다. 예를 들어, 1D 레이어를 2D로 확장하거나 그 반대의 작업을 수행할 수 있습니다.

### Word Embedding 하위 모듈

`word_embedding` 모듈은 Word2Vec, FastText와 같은 사전 훈련된 임베딩을 사용하는 임베딩 레이어를 제공합니다. 이 모듈은 자연어 처리 작업에서 임베딩을 쉽게 사용할 수 있도록 합니다.

- **Word2VecEmbedding**: Gensim에서 제공하는 사전 훈련된 Word2Vec 임베딩을 로드하여 사용하는 클래스입니다.
- **FastTextEmbedding**: Gensim에서 제공하는 사전 훈련된 FastText 임베딩을 로드하여 사용하는 클래스입니다.

## 사용 예시

### AutoEncoder 및 VAE 사용
`AutoEncoder` 클래스와 이를 확장한 `VAE`(Variational AutoEncoder)는 이미지 데이터를 인코딩하고 디코딩하여 재구성하는 데 사용됩니다. U-Net 스타일의 스킵 연결을 지원하여 더욱 효과적인 이미지 재구성이 가능합니다.

```python
from zae_engine.models.builds import AutoEncoder
from zae_engine.nn_night.blocks import UNetBlock

# AutoEncoder 모델 생성
model = AutoEncoder(
    block=UNetBlock,
    ch_in=3,
    ch_out=1,
    width=32,
    layers=[2, 2, 2, 2],
    skip_connect=True
)
```

### ResNet 사용
ResNet은 이미지 인식 작업에서 널리 사용되는 네트워크입니다. `resnet18`, `resnet34`, `resnet50` 등 다양한 버전을 사용할 수 있으며, SE 모듈이나 CBAM 모듈을 주입하여 성능을 향상시킬 수 있습니다.

```python
from zae_engine.models.foundations import resnet18

# ResNet-18 모델 생성
model = resnet18(pretrained=True)
```

### Transformer 사용
`TransformerBase` 클래스는 인코더-디코더 아키텍처를 지원하며, 번역과 같은 시퀀스-투-시퀀스 작업에서 사용할 수 있습니다.

```python
from zae_engine.models.builds import TransformerBase

# Transformer 모델 생성
transformer = TransformerBase(
    encoder_embedding=encoder_emb_layer,
    decoder_embedding=decoder_emb_layer,
    encoder=encoder_module,
    decoder=decoder_module
)
```

