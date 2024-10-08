# 데이터 서브패키지

`zae-engine`의 `data` 서브패키지는 PyTorch를 사용한 딥러닝 프로젝트에서 효율적인 데이터 처리를 위해 필수적인 유틸리티들을 제공합니다. 이 서브패키지는 PyTorch의 `Dataset`과 `collate` 기능을 확장하여 다양한 데이터 형식을 손쉽게 관리할 수 있도록 합니다. 특히 대규모 데이터 처리에 적합합니다.

## 개요

`data` 서브패키지는 주로 다음과 같은 구성 요소로 이루어져 있습니다:

- **Dataset 확장**: Parquet과 같은 비표준 데이터 형식을 손쉽게 다룰 수 있도록 커스텀 데이터셋을 제공합니다.
- **DataLoader를 위한 CollateBase**: 여러 맞춤형 함수를 사용해 데이터 전처리 및 배치를 관리하는 기본 클래스입니다. 이를 PyTorch의 `DataLoader`의 `collate_fn`으로 사용할 수 있습니다.

## 주요 기능

### 1. ParquetDataset
`ParquetDataset` 클래스는 PyTorch의 `Dataset`을 확장하여 Parquet 파일을 처리할 수 있도록 한 커스텀 데이터셋입니다. 이 데이터셋은 다음과 같은 기능을 제공합니다:

- **캐싱**: 여러 Parquet 파일을 메모리에 로드하여 효율적인 접근을 지원하며, 캐시 크기를 사용자가 설정할 수 있습니다.
- **셔플링**: 인덱스를 무작위로 셔플링할 수 있는 옵션을 제공하여 랜덤화된 데이터 접근을 지원합니다.
- **컬럼 선택**: Parquet 파일에서 특정 컬럼만 선택해서 로드할 수 있어 필요한 데이터만 사용할 수 있습니다.

#### 사용 예시
```python
from torch.utils.data import DataLoader
from zae_engine.data.dataset import ParquetDataset

# ParquetDataset 초기화 예시
parquet_paths = ["data/file1.parquet", "data/file2.parquet"]
dataset = ParquetDataset(parquet_path=parquet_paths, fs=None, columns=("col1", "col2"))

# DataLoader와 함께 사용
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

`ParquetDataset`은 PyTorch의 `DataLoader`와의 통합을 통해 대규모 Parquet 데이터를 딥러닝 작업에 쉽게 활용할 수 있게 해줍니다.

### 2. CollateBase
`CollateBase` 클래스는 배치 데이터를 수집하면서 복잡한 전처리 작업을 처리하도록 설계되었습니다. 여러 전처리 함수를 배치에 적용하는 모듈식 방법을 제공합니다.

- **유연한 초기화**: `OrderedDict` 또는 함수 리스트를 사용하여 초기화할 수 있으며, 각 배치에 순차적으로 적용됩니다.
- **배치 처리**: 딕셔너리의 리스트를 배치 딕셔너리로 변환하여 PyTorch `DataLoader`의 `collate_fn`과 호환되도록 합니다.
- **입출력 구조 검사**: 등록된 전처리 함수들이 샘플 데이터의 입출력 구조를 유지하는지 자동으로 검사하여 데이터 불일치 가능성을 줄입니다.

#### 사용 예시
```python
from collections import OrderedDict
from zae_engine.data.collate import CollateBase

def custom_fn1(batch):
    # 사용자 정의 전처리 함수
    return batch

def custom_fn2(batch):
    # 또 다른 사용자 정의 전처리 함수
    return batch

functions = OrderedDict([('fn1', custom_fn1), ('fn2', custom_fn2)])
collator = CollateBase(x_key=['x'], y_key=['y'], aux_key=['aux'], functions=functions)

# DataLoader와 CollateBase 사용
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
```

`CollateBase`를 사용하면 각 배치에 적용할 일련의 변환을 정의하여 데이터 전처리를 매우 모듈화하고 사용자 정의할 수 있습니다.

### 3. CollateBase를 위한 전처리 함수
`CollateBase` 외에도 `data/collate` 모듈에는 `CollateBase`의 기능을 확장하기 위해 사용할 수 있는 여러 전처리 클래스가 포함되어 있습니다. 이러한 클래스들은 다음과 같습니다:

- **UnifiedChunker**: 입력 데이터를 차원에 따라 청크로 분할하며, 1D 및 2D 텐서를 모두 지원합니다. 시퀀스 데이터를 작은 조각으로 나누어야 할 때 유용합니다.
- **Chunk**: 배치 내 텐서를 특정 청크 크기로 재구성합니다.
- **HotEncoder**: 레이블을 원-핫 인코딩 형식으로 변환하여 분류 작업을 쉽게 수행할 수 있습니다.
- **SignalFilter**: 신호 데이터를 필터링(예: 밴드패스, 로우패스)하여 시계열 또는 센서 데이터 전처리에 유용합니다.
- **Spliter**: 배치 내 신호를 중첩된 세그먼트로 분할하여 겹쳐진 시퀀스를 생성합니다.
- **SignalScaler**: MinMaxScaler를 사용하여 신호 데이터를 스케일링하고, 값의 범위를 일관되게 유지합니다.

#### 전처리 함수 사용 예시
```python
from zae_engine.data.collate import CollateBase, HotEncoder, SignalFilter

# 전처리 함수 정의
encoder = HotEncoder(n_cls=10)
filter_fn = SignalFilter(fs=100.0, method='bandpass', lowcut=0.5, highcut=30.0)

collator = CollateBase(functions=[encoder, filter_fn])

# 사용자 정의 collator와 DataLoader 사용
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
```
이러한 전처리 함수들은 사용자가 입력 데이터에 대해 복잡한 변환을 유연하게 처리할 수 있게 하며, `CollateBase`를 배치 처리 과정에서 강력한 도구로 만들어 줍니다.

## 요약
`zae-engine`의 `data` 서브패키지는 PyTorch의 데이터 처리 도구를 확장하여 다음을 제공합니다:
- 대규모 데이터 형식을 처리하기 위한 `ParquetDataset`과 같은 커스텀 데이터셋.
- 데이터 배치 전처리를 위한 모듈식 `CollateBase` 클래스.
- 청크 처리, 인코딩, 필터링과 같은 작업을 위한 여러 내장 전처리 함수.

이러한 도구들은 특히 대규모 또는 특수한 데이터 형식을 위한 데이터 준비 과정을 간소화하며, PyTorch의 `DataLoader`와 원활하게 통합되는 모듈식 전처리 워크플로우를 지원합니다.

