# Operation 서브패키지

**`operation` 서브패키지는 알고리즘 기반의 복잡한 연산을 지원하여 다양한 데이터 처리와 변형을 쉽게 할 수 있도록 돕습니다.** 이 서브패키지에는 다양한 데이터 변형 및 분석 기능을 제공하는 여러 클래스와 함수가 포함되어 있습니다. 이들을 통해 딥러닝 모델에서 사용하는 입력 데이터를 더 효율적이고 유연하게 처리할 수 있습니다.

## 주요 기능

### 1. MorphologicalLayer

`MorphologicalLayer` 클래스는 입력된 1차원 텐서에 대해 다양한 형태학적 연산(모폴로지 연산)을 수행할 수 있는 레이어입니다. 이 레이어는 팽창(dilation)과 침식(erosion) 같은 연산을 통해 데이터의 특정 패턴을 강조하거나 필터링하는 데 사용됩니다.

- **매개변수**
  - `ops`: 연산의 순서를 나타내는 문자열로, 'c'는 closing(팽창 후 침식), 'o'는 opening(침식 후 팽창)을 의미합니다.
  - `window_size`: 각 연산에 대한 윈도우 크기 리스트입니다.
- **사용 예시**
  ```python
  morph_layer = MorphologicalLayer(ops='co', window_size=[5, 3])
  output = morph_layer(input_tensor)
  ```

### 2. arg_nearest

`arg_nearest` 함수는 주어진 정렬된 배열에서 특정 값에 가장 가까운 값을 찾아주는 함수입니다. 입력 배열이 오름차순으로 정렬되어 있어야 하며, 이 함수는 참조 값과 가장 가까운 요소의 인덱스와 값을 반환합니다.

- **매개변수**
  - `arr`: 정렬된 입력 배열(NumPy 배열 또는 PyTorch 텐서).
  - `value`: 참조 값.
  - `return_value`: 인덱스와 값을 함께 반환할지 여부(기본값은 True).
- **사용 예시**
  ```python
  import numpy as np
  arr = np.array([1, 3, 5, 7, 9])
  index, nearest_value = arg_nearest(arr, 6)
  ```

### 3. RunLengthCodec

`RunLengthCodec` 클래스는 데이터 압축과 복원에 사용되는 런-길이 인코딩(Run-Length Encoding, RLE)을 구현한 클래스입니다. 주어진 데이터 시퀀스를 압축된 형태로 변환하거나, 압축된 데이터를 원래의 형태로 복원하는 기능을 제공합니다.

- **매개변수**
  - `tol_merge`: 인접한 런을 병합할 때 허용할 간격의 크기(기본값은 20).
  - `remove_incomplete`: 불완전한 런을 제거할지 여부(기본값은 False).
  - `merge_closed`: 인접한 런을 병합할지 여부(기본값은 False).
  - `base_class`: 런에서 제외할 기본 클래스 값(기본값은 0).
- **사용 예시**
  ```python
  codec = RunLengthCodec(tol_merge=10)
  encoded_runs = codec.encode([1, 1, 0, 0, 2, 2, 2], sense=2)
  decoded_data = codec.decode(encoded_runs)
  ```

## 활용 사례

- **형태학적 필터링**: `MorphologicalLayer`를 사용하여 입력 데이터를 전처리하거나 특정 패턴을 강조하는 데 사용할 수 있습니다. 예를 들어, 신호 데이터에서 잡음을 제거하거나 특징을 강조할 수 있습니다.
- **근접 값 찾기**: `arg_nearest` 함수를 사용하여 특정 값에 가장 가까운 데이터를 빠르게 찾을 수 있습니다. 이는 시간 축 상에서 특정 이벤트와 가까운 다른 이벤트를 찾는 데 유용합니다.
- **데이터 압축 및 복원**: `RunLengthCodec`은 데이터를 효율적으로 압축하고, 나중에 복원할 수 있어 메모리 절약과 빠른 처리를 가능하게 합니다. 예를 들어, 동일한 값이 연속적으로 나타나는 긴 시퀀스를 효율적으로 표현할 수 있습니다.

## 결론

`operation` 서브패키지는 딥러닝 모델의 데이터 처리 과정에서 복잡한 연산을 간단하게 수행할 수 있도록 설계되었습니다. 이 서브패키지를 통해 데이터 전처리 및 변형 작업을 더욱 간편하고 효율적으로 수행할 수 있습니다.

