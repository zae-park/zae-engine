# Data Collate

## 개요

**zae_engine**는 머신러닝 및 딥러닝 워크플로우를 위한 강력한 데이터 전처리 라이브러리입니다. 이 라이브러리는 데이터 준비 과정을 간소화하고, 일관된 전처리 파이프라인을 제공하여 모델의 성능을 극대화하는 데 도움을 줍니다. **CollateBase**는 zae_engine의 핵심 모듈 중 하나로, 텐서 데이터를 청크로 분할하고, 스케일링하며, 필터링하는 다양한 전처리 작업을 수행합니다.

## 주요 기능

- **통합 청크 분할 (Unified Chunking):** 1D 및 2D 텐서를 관리 가능한 청크로 유연하게 분할.
- **신호 필터링 (Signal Filtering):** 저역통과, 고역통과, 대역통과, 대역차단 등 다양한 신호 처리 필터 적용.
- **신호 스케일링 (Signal Scaling):** Min-Max 스케일링을 사용하여 데이터 정규화.
- **유연한 콜레이션 (Flexible Collation):** 여러 전처리 모듈을 손쉽게 통합하여 데이터 파이프라인 구성.
- **포괄적인 테스트 (Comprehensive Testing):** 각 모듈의 신뢰성과 정확성을 보장하는 강력한 단위 테스트.


## 사용 방법

### 기본 예제

아래는 **CollateBase**를 사용하여 텐서 데이터를 청크로 분할하고, 스케일링하며, 필터링하는 간단한 예제입니다.

```python
import torch
from zae_engine.data.collate import CollateBase, UnifiedChunker, SignalFilter, SignalScaler

# 전처리 모듈 초기화
chunker = UnifiedChunker(chunk_size=3, overlap=1)
scaler = SignalScaler()
filter = SignalFilter(fs=1000, method='lowpass', cutoff=10)

# CollateBase 인스턴스 생성 및 모듈 추가
collate = CollateBase(functions=[chunker, scaler, filter])

# 예제 데이터 배치
batch = {
    'x': torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32),  # Shape: (8,)
    'y': torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32),  # Shape: (8,)
    'fn': 'sample_fn'
}

# 전처리 적용
processed_batch = collate(batch)

print(processed_batch)
```

### 고급 구성
전처리 파이프라인을 사용자 정의하여 청크 크기와 오버랩을 조정하고, 다양한 필터를 적용할 수 있습니다.

```python
import torch
from zae_engine.data.collate import CollateBase, UnifiedChunker, SignalFilter, SignalScaler

# 전처리 모듈 초기화 (커스텀 파라미터 사용)
chunker = UnifiedChunker(chunk_size=5, overlap=2)
scaler = SignalScaler()
filter = SignalFilter(fs=2000, method='bandpass', lowcut=5, highcut=15)

# CollateBase 인스턴스 생성 및 모듈 추가
collate = CollateBase(functions=[filter, scaler, chunker])

# 예제 데이터 배치
batch = {
    'x': torch.tensor([0.5, 1.2, 3.4, 2.2, 5.5, 6.1, 7.8, 8.9, 9.0, 10.1], dtype=torch.float32),  # Shape: (10,)
    'y': torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32),  # Shape: (10,)
    'fn': ['advanced_sample']
}

# 전처리 적용
processed_batch = collate(batch)

print(processed_batch)

```

## API 참조

### CollateBase

`CollateBase`는 여러 전처리 모듈을 순차적으로 적용하여 데이터 배치를 처리하는 클래스입니다.

**초기화 파라미터:**

- `modules` (List[Callable]): 전처리 모듈의 리스트. 순서대로 적용됩니다.

**메소드:**

- `__call__(batch: Dict[str, Any]) -> Dict[str, Any]`: 지정된 전처리 모듈을 순차적으로 배치에 적용합니다.

**예제:**

```python
from zae_engine.data.collate import CollateBase, UnifiedChunker, SignalFilter

chunker = UnifiedChunker(chunk_size=4, overlap=2)
filter = SignalFilter(fs=1000, method='highpass', cutoff=5)

collate = CollateBase(functions=[filter, chunker])

processed_batch = collate(batch)

```




