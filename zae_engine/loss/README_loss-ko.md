# 손실 함수 서브패키지

`zae-engine`의 `loss` 서브패키지는 다양한 딥러닝 작업에 사용할 수 있는 여러 손실 함수를 제공합니다. 분류와 회귀 작업에 사용되는 기본 손실 함수부터 각도 기반 마진 및 IoU 기반의 고급 손실 함수까지 포함하고 있습니다.

## 개요

`loss` 서브패키지에는 다음과 같은 손실 함수들이 포함되어 있습니다:

- **cross_entropy**: 로짓을 사용하여 이진 교차 엔트로피 손실을 계산합니다. 주로 분류 작업에 사용됩니다.
  ```python
  from zae_engine.loss import cross_entropy
  loss = cross_entropy(logit, y_hot, class_weights)
  ```
- **compute_gram_matrix**: 벡터 배치의 Gram 행렬을 계산합니다. 스타일 전이 또는 정규화 목적으로 사용할 수 있습니다.
  ```python
  from zae_engine.loss import compute_gram_matrix
  gram_matrix = compute_gram_matrix(batch)
  ```
- **ArcFaceLoss**: ArcFace 손실을 구현하며, 각도 마진을 도입하여 분류 모델의 변별력을 향상시킵니다. 특히 얼굴 인식 작업에 유용합니다.
  ```python
  from zae_engine.loss import ArcFaceLoss
  loss_fn = ArcFaceLoss(in_features=512, out_features=10)
  loss = loss_fn(features, labels)
  ```
  - **매개변수**: `in_features`, `out_features`, `s` (입력의 노름), `m` (마진).
  - 이 손실 함수는 Jiankang Deng 등이 저술한 "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" 논문을 기반으로 합니다.
- **mIoU**: 평균 교차합집합(mean Intersection over Union)을 계산하여 객체 분할의 정확도를 평가하는 데 사용됩니다.
  ```python
  from zae_engine.loss import mIoU
  score = mIoU(pred, true)
  ```
- **IoU**: 객체 분할 작업에서 교차합집합(Intersection over Union)을 계산합니다.
  ```python
  from zae_engine.loss import IoU
  score = IoU(pred, true)
  ```
- **GIoU**: 일반화된 교차합집합(Generalized IoU)을 계산하여 객체 검출 작업에 유용합니다. GIoU는 겹치지 않는 바운딩 박스를 더 잘 처리할 수 있도록 IoU를 확장한 것입니다.
  ```python
  from zae_engine.loss import GIoU
  score = GIoU(true_onoff, pred_onoff)
  ```
- **mse**: 평균 제곱 오차(Mean Squared Error)를 계산하여 예측값과 실제 값 사이의 평균 제곱 차이를 측정합니다. 기본적인 회귀 손실 함수입니다.
  ```python
  from zae_engine.loss import mse
  loss = mse(true, predict)
  ```
  - MSE는 회귀 작업에서 예측 정확도를 측정하는 데 주로 사용됩니다.

## 요약
`zae-engine`의 `loss` 서브패키지는 다양한 딥러닝 작업을 지원하는 포괄적인 손실 함수 모음을 제공합니다. 이 손실 함수들은 다음과 같습니다:
- **기본 손실 함수**: 분류(`cross_entropy`) 및 회귀(`mse`) 작업을 위한 손실 함수.
- **고급 각도 마진 손실**: 얼굴 인식 및 변별력이 필요한 작업을 위한 손실 함수 (`ArcFaceLoss`).
- **IoU 기반 손실**: 객체 분할 및 검출 작업을 위한 손실 함수 (`mIoU`, `IoU`, `GIoU`).

이러한 손실 함수들은 사용자가 다양한 딥러닝 응용 프로그램에서 효과적인 학습 전략을 쉽게 구현할 수 있도록 도와줍니다.

