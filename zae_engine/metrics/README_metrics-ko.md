# 메트릭 서브패키지

`zae-engine`의 `metrics` 서브패키지는 딥러닝 모델의 성능을 평가하기 위한 다양한 정량적 지표를 제공합니다. 이 서브패키지에서는 모델 출력과 실제 라벨을 비교하여 다양한 성능 지표를 계산할 수 있으며, 이를 통해 모델의 성능을 객관적으로 평가하고 개선 방향을 모색할 수 있습니다.

## 개요

`metrics` 서브패키지에는 다음과 같은 메트릭들이 포함되어 있습니다:

- **BijectiveMetrix**: 예측된 시퀀스와 실제 시퀀스 사이의 상호적인 매핑을 통해 일치 행렬(confusion matrix)을 계산합니다. 서브패키지 내의 다른 함수들과 함께 사용하여 F1 점수와 정확도를 확인할 수 있습니다.
  ```python
  from zae_engine.metrics import BijectiveMetrix
  metric = BijectiveMetrix(prediction, label, num_classes=3)
  print(metric.bijective_f1)
  ```
  - **매개변수**: `prediction`, `label`, `num_classes`, `th_run_length` (기본값: 2).
- **confusion_matrix**: 분류 문제의 예측 결과와 실제 라벨을 비교하여 혼동 행렬(confusion matrix)을 계산합니다. 이를 통해 클래스 간의 오차 분포를 분석할 수 있습니다.
  ```python
  from zae_engine.metrics import confusion_matrix
  conf_mat = confusion_matrix(y_hat, y_true, num_classes=3)
  ```
- **accuracy**: 예측값과 실제 라벨의 일치 비율을 계산하여 모델의 정확도를 평가합니다.
  ```python
  from zae_engine.metrics import accuracy
  acc = accuracy(true, predict)
  ```
- **f_beta**: F-beta 점수를 계산합니다. 이 점수는 재현율과 정밀도 사이의 균형을 맞추는 데 사용되며, `beta` 값을 조정하여 재현율 또는 정밀도를 더 강조할 수 있습니다.
  ```python
  from zae_engine.metrics import f_beta
  f_beta_score = f_beta(pred, true, beta=1.0, num_classes=3, average='micro')
  ```
  - **매개변수**: `beta` 값으로 재현율과 정밀도의 비중을 설정할 수 있으며, `average` 옵션으로 `micro` 또는 `macro` 평균을 선택할 수 있습니다.
- **miou**: 평균 교차합집합(mean Intersection over Union, mIoU)을 계산하여 객체 분할 성능을 평가합니다.
  ```python
  from zae_engine.metrics import miou
  score = miou(img1, img2)
  ```
- **giou**: 일반화된 교차합집합(Generalized IoU)을 계산합니다. GIoU는 객체 검출에서 겹치지 않는 바운딩 박스를 잘 처리할 수 있도록 IoU를 확장한 것입니다.
  ```python
  from zae_engine.metrics import giou
  score = giou(img1, img2)
  ```
- **rms**: 신호의 제곱 평균 루트(Root Mean Square, RMS)를 계산하여 신호의 크기를 평가합니다.
  ```python
  from zae_engine.metrics import rms
  rms_value = rms(signal)
  ```
- **signal_to_noise**: 신호 대 잡음비(Signal-to-Noise Ratio, SNR)를 계산하여 신호와 잡음의 비율을 평가합니다.
  ```python
  from zae_engine.metrics import signal_to_noise
  snr_value = signal_to_noise(signal, noise)
  ```
- **peak_signal_to_noise**: 피크 신호 대 잡음비(PSNR)를 계산합니다. PSNR은 신호의 피크 값과 잡음의 비율을 측정하는 데 사용됩니다.
  ```python
  from zae_engine.metrics import peak_signal_to_noise
  psnr_value = peak_signal_to_noise(signal, noise)
  ```

## 요약
`zae-engine`의 `metrics` 서브패키지는 모델의 성능을 정량적으로 평가하기 위한 다양한 메트릭들을 제공합니다. 이 메트릭들은 다음과 같습니다:
- **혼동 행렬 기반 메트릭**: 클래스 간의 예측 성능을 분석할 수 있는 `confusion_matrix`와 `BijectiveMetrix`.
- **기본 정확도 메트릭**: 분류 정확도를 평가하기 위한 `accuracy`.
- **F-beta 점수**: 재현율과 정밀도의 균형을 맞추기 위한 `f_beta`.
- **IoU 기반 메트릭**: 객체 분할 및 검출 성능을 평가하기 위한 `miou`, `giou`.
- **신호 분석 메트릭**: 신호의 크기와 잡음 비율을 평가하기 위한 `rms`, `signal_to_noise`, `peak_signal_to_noise`.

이러한 메트릭들은 사용자가 딥러닝 모델의 성능을 전반적으로 평가하고 개선하는 데 큰 도움을 줄 수 있습니다.

