# nn_night 블록 서브패키지 개요

**nn_night**의 블록 서브패키지는 딥러닝 네트워크에서 사용되는 다양한 블록을 제공합니다. 이 블록들은 기본적인 잔차 블록에서부터 고급 주의 메커니즘 및 U-Net 아키텍처용 블록까지 포함하고 있어, 사용자가 다양한 아키텍처와 작업에 맞게 조합하여 사용할 수 있습니다. 이 README는 주요 블록들과 그들의 사용법을 설명합니다.

## 주요 구성 요소

### 1. BasicBlock & Bottleneck

- **BasicBlock**과 **Bottleneck**은 ResNet 아키텍처에서 널리 사용되는 두 가지 잔차(residual) 블록 유형입니다.
- **BasicBlock**은 간단한 잔차 연결을 통해 입력과 출력을 더하여 학습을 안정화하고 성능을 향상시킵니다. 주로 ResNet-18, ResNet-34 같은 얕은 네트워크에서 사용됩니다.
- **Bottleneck**은 더 깊은 네트워크에서의 효율적인 학습을 위해 채널 축소 및 확장을 수행하며, ResNet-50, ResNet-101 등의 깊은 모델에서 사용됩니다.

### 2. SE1d & CBAM1d

- **SE1d (Squeeze-and-Excitation)** 모듈은 채널 간의 상호작용을 학습하여 중요한 특성을 강조합니다. 이를 통해 네트워크의 표현력을 높일 수 있습니다.
- **CBAM1d (Convolutional Block Attention Module)**는 SE 모듈에 공간적 주의 메커니즘을 추가하여 채널과 공간 모두에서 중요한 정보를 강조합니다. 이 모듈은 이미지나 신호 처리와 같이 입력 데이터의 공간적/채널적 특징을 모두 고려해야 하는 상황에서 유용합니다.

### 3. UNetBlock

- **UNetBlock**은 U-Net 아키텍처에서 사용되기 위해 설계된 잔차 블록입니다. 이 블록은 잔차 연결을 통해 학습을 안정화하고, U-Net의 특성인 업샘플링과 다운샘플링 단계에서 효과적으로 작동하도록 조정되었습니다.
- U-Net 아키텍처는 주로 의료 영상 세분화와 같은 컴퓨터 비전 작업에 사용됩니다.

## 사용 예시

`nn_night.blocks` 서브패키지는 다양한 딥러닝 아키텍처를 구현하기 위해 필요한 여러 블록들을 제공합니다. 예를 들어, ResNet 스타일 네트워크를 구축하거나 U-Net과 같은 복잡한 모델을 설계할 때, 이 블록들을 유연하게 사용할 수 있습니다.

```python
import torch
import torch.nn as nn
from nn_night.blocks import BasicBlock, SE1d, UNetBlock

# BasicBlock 사용 예시
basic_block = BasicBlock(ch_in=64, ch_out=64)
input_tensor = torch.randn(1, 64, 32, 32)  # 예시 입력 텐서
output = basic_block(input_tensor)

# SE1d 사용 예시
se_block = SE1d(ch_in=64)
input_tensor_1d = torch.randn(1, 64, 128)  # 1D 입력 텐서
output_1d = se_block(input_tensor_1d)

# UNetBlock 사용 예시
unet_block = UNetBlock(ch_in=64, ch_out=64)
unet_output = unet_block(input_tensor)
```
위 예시는 `BasicBlock`, `SE1d`, `UNetBlock`을 각각 사용하여 입력 텐서를 처리하는 방법을 보여줍니다. 이를 통해 네트워크의 특정 부분을 쉽게 구축할 수 있습니다.

## 요약

`nn_night.blocks` 서브패키지는 딥러닝 아키텍처에서 사용되는 다양한 블록을 제공하여 네트워크 구축을 용이하게 합니다. 이러한 블록들은 딥러닝 연구와 최신 모델 설계에서 자주 사용되는 개념들을 반영하고 있으며, 사용자가 필요에 따라 적절히 조합하여 강력한 모델을 설계할 수 있도록 돕습니다.

