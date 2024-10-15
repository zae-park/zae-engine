# nn\_night

**nn\_night**는 PyTorch의 nn 서브패키지와 유사한 기능을 제공하는 개인적으로 개발한 서브패키지입니다. 이 패키지는 딥러닝의 다양한 논문을 참고하여 여러 레이어와 구조를 구현하였으며, 앞으로도 계속해서 최신 연구에서 영감을 받은 새로운 아이디어를 추가할 예정입니다.

## 주요 개요

이 패키지는 다양한 신경망 레이어와 구조를 포함하고 있어, 사용자들이 자신만의 맞춤형 모델을 구축하는 데 도움을 줄 수 있습니다. 레이어들은 모두 PyTorch 모듈을 확장하여 구현되었으며, 이를 통해 기존의 PyTorch 학습 루틴과 손쉽게 통합할 수 있습니다.

## 구현된 구성 요소

1. **ClippedReLU**

   - ReLU 함수의 변형으로 입력 텐서를 [lower, upper] 범위로 제한합니다. 이를 통해 활성화 함수에서 출력 값을 제어할 수 있습니다.

2. **GumbelSoftMax**

   - 범주형 문제를 해결하기 위한 Gumbel-Softmax 트릭을 구현한 함수로, 신경망에서 범주형 변수를 미분 가능한 방식으로 샘플링할 수 있게 해줍니다.

3. **Additional**

   - 여러 입력 텐서를 합산하는 추가 연결 모듈입니다. 각 입력 텐서를 해당 모듈에 통과시켜, 그 출력값들을 더합니다.

4. **DynOPool**

   - Gumbel-Softmax 트릭을 사용하여 이산적인 풀링 비율을 학습하는 동적 풀링 레이어입니다. 학습 과정에서 적응형 풀링을 통해 성능을 최적화합니다.

5. **Inv1d (Involution Layer)**

   - Involution 기법을 사용한 1D 레이어로, 비전 인식 문제에 대한 새로운 접근 방식인 Involution을 구현하였습니다. 이는 입력 텐서의 특성을 효과적으로 반영하는 특수한 필터링 과정을 수행합니다.

6. **SinusoidalPositionalEncoding, LearnablePositionalEncoding, RotaryPositionalEncoding, RelativePositionalEncoding, AdaptivePositionalEncoding**

   - 다양한 형태의 위치 인코딩을 제공하여 입력 시퀀스의 위치 정보를 인코딩합니다. 이를 통해 시퀀스 데이터를 다루는 모델에서 위치 정보를 효과적으로 반영할 수 있습니다.

7. **Residual**

   - 입력 텐서에 여러 모듈의 출력을 더하여 잔차 연결을 구현하는 레이어입니다. ResNet과 유사한 방식으로 신경망의 기울기 소실 문제를 완화할 수 있습니다.

8. **SKConv1D (Selective Kernel Convolution)**

   - 선택적 커널 네트워크를 구현한 1D 레이어로, 여러 커널 크기를 사용하여 다양한 특성을 학습하고, 이를 선택적으로 결합하는 과정을 포함합니다.

## 사용 예시

사용자는 nn\_night에서 제공하는 다양한 레이어들을 조합하여 자신만의 신경망 모델을 만들 수 있습니다. 예를 들어, **ClippedReLU**나 **DynOPool** 같은 레이어를 사용하여 기존의 모델에 추가적인 비선형성과 동적 학습 기능을 도입할 수 있습니다.

```python
import torch
import torch.nn as nn
from nn_night import ClippedReLU, DynOPool

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.act1 = ClippedReLU(upper=1.0, lower=0.0)
        self.pool = DynOPool()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool(x)
        return x

model = CustomModel()
input_tensor = torch.randn(8, 16, 100)
output = model(input_tensor)
print(output.shape)
```

위 예시는 **ClippedReLU**와 **DynOPool**을 사용하여 간단한 1D 컨볼루션 네트워크를 구축하는 예입니다. 이처럼 nn\_night에서 제공하는 레이어들을 사용하여 신경망을 손쉽게 확장하고 다양한 특성을 학습할 수 있습니다.

## 요약
nn_night 서브패키지는 최근 딥러닝 연구의 발전에 영감을 받아 신경망 설계를 위한 강력하고 유연한 컴포넌트들을 제공합니다. 이 모듈들은 동적이고, 적응적이며, 정교한 메커니즘을 활용하는 복잡한 아키텍처를 구성하는 데 사용될 수 있으며, 현대 딥러닝 모델의 한계를 넓히기 위한 도구를 제공합니다.