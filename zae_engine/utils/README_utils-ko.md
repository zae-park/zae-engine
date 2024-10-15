# zae-engine: Utils 서브패키지

**utils** 서브패키지는 다양한 유틸리티 기능을 제공하여 데이터 입출력, 변환, 디버깅 등을 손쉽게 할 수 있도록 돕습니다. 이 서브패키지는 `io`와 `decorators` 두 가지 주요 모듈로 구성되어 있으며, 각 모듈은 다양한 상황에서 사용할 수 있는 기능들을 제공합니다.

## 주요 기능

### I/O 모듈

- **`image_from_url`**: 주어진 URL에서 이미지를 다운로드하여 파일로 저장하거나 PIL 이미지 객체로 반환합니다. 이 기능은 이미지 데이터를 원격에서 손쉽게 불러올 수 있도록 도와줍니다.

- **`example_ecg`**: LUDB 데이터셋에서 10초간의 ECG 신호와 주석을 불러오는 기능을 제공합니다. 기본적으로 10초간의 전체 데이터를 반환하지만, 특정 비트를 지정하여 해당 구간의 신호와 정보를 반환할 수도 있습니다.

- **`example_mri`**: NIfTI 파일 형식의 MRI 데이터를 불러옵니다. 이 함수는 4차원 MRI 스캔 데이터를 불러와서 `ArrayProxy` 객체로 반환합니다.

### Decorators 모듈

`decorators` 모듈은 여러 데코레이터 함수를 제공하여 함수나 메서드의 기능을 확장할 수 있도록 돕습니다.

- **`np2torch`**: numpy 배열을 PyTorch 텐서로 변환하는 데코레이터입니다. 지정된 키나 인자를 기준으로 변환을 수행하여 데이터 타입의 일관성을 유지할 수 있습니다.

- **`torch2np`**: PyTorch 텐서를 numpy 배열로 변환하는 데코레이터입니다. 모델 예측 후 결과를 numpy로 변환하거나, 특정 키를 기준으로 변환할 때 유용합니다.

- **`tictoc`**: 함수의 실행 시간을 측정하고 출력하는 데코레이터입니다. 함수 또는 메서드가 얼마나 오래 걸리는지 쉽게 추적할 수 있도록 도와줍니다.

- **`shape_check`**: 여러 인자의 shape이 동일한지 확인하는 데코레이터입니다. 함수 호출 시 인자의 형태가 일치하는지 검증하여 오류를 방지할 수 있습니다.

## 사용 예시

아래는 각 모듈에서 제공하는 기능을 사용하는 간단한 예시입니다.

### 이미지 다운로드 예시
```python
from zae_engine.utils import image_from_url

# 이미지 다운로드 및 저장
image_from_url('https://example.com/image.png', 'downloaded_image.png')

# 이미지 다운로드 및 PIL 이미지로 반환
img = image_from_url('https://example.com/image.png')
img.show()
```

### Decorator 사용 예시
```python
from zae_engine.utils.decorators import np2torch, tictoc
import numpy as np
import torch

@np2torch(torch.float32, 'data')
@tictoc
def process_data(data):
    # numpy 배열이 PyTorch 텐서로 변환되어 처리됩니다.
    return data * 2

# 사용 예시
data = np.array([1, 2, 3])
result = process_data({'data': data})
print(result)
```

## 요약

**utils** 서브패키지는 데이터 입출력 및 다양한 유틸리티 기능을 제공합니다. `io` 모듈은 이미지와 의료 데이터를 불러오는 기능을 제공하며, `decorators` 모듈은 데이터 변환, 실행 시간 측정, 형태 검사 등의 기능을 데코레이터 형태로 제공합니다. 이를 통해 사용자는 모델 학습 및 평가 과정에서 반복적으로 사용하는 작업들을 보다 간단하고 효율적으로 수행할 수 있습니다.