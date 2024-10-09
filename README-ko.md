<div align="center">

<p align="center">
  <img src="assets/img/spinning_ascii_donut.gif" style="border-radius: 20px">
  <br />
</p>

# zae-engine
[ <a href="./README.md">English</a> ]
<span style="display: inline-block; width: 20px;"></span>
[ <a href="./README-ko.md">Korean</a> ]
</div>

<p align="center">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/build_test.yml/badge.svg">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/document_deploy.yml/badge.svg" alt="https://zae-park.github.io/zae-engine">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/unittest_badge.yml/badge.svg">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/wandb_test.yml/badge.svg">
    </br>
    <img src="https://codecov.io/gh/zae-park/zae-engine/graph/badge.svg?token=4BENXZJHPF">
    <img src="https://img.shields.io/pypi/pyversions/zae-engine.svg" alt="https://pypi.python.org/pypi/zae-engine/">
    <img src="https://img.shields.io/pypi/v/zae-engine.svg" alt="https://pypi.python.org/pypi/zae-engine/">
    <img src="https://img.shields.io/pypi/dm/zae-engine.svg" alt="https://pypi.python.org/pypi/zae-engine/">
  <br />
</p>

# Accelerate AI project !

`zae-engine`은 PyTorch, TensorFlow, Flax와 같은 딥러닝 프레임워크들을 정리한 package입니다. 
AI 프로젝트의 모든 단계를 표준화하여 반복적인 코드를 줄이고 데이터 과학을 가속화합니다.
또, 코드의 단순성과 유연성을 높여 복잡한 딥러닝 프로젝트를 쉽게 관리하고 유지 보수할 수 있습니다.


## Advantages over unstructured PyTorch
- 빈번하게 사용되는 boilerplate를 정리하여 코드의 가독성을 높입니다.
- 까다로운 엔지니어링을 추상화하여 `zae-engine`이 처리하므로 실수를 줄일 수 있습니다.
- 최소한의 코드 수정으로 고급 엔지니어링으로부터 데이터 과학을 디커플링합니다. (ex. CPU, Apple Silicon, CUDA 지원 및 멀티 GPU 전환)
- 다양한 구현체들을 일반화하여 유연성을 높입니다.
- 모든 유연성을 유지하면서(`LightningModules`가 여전히 PyTorch 모듈인 것처럼), 반복적인 보일러플레이트 코드를 제거합니다.
- 체크포인트 저장, 로깅 등의 확장을 간단하게 제공합니다.
- 수십 가지의 유용하고 효과적인 모델, 레이어, 툴이 통합되어 있습니다.

## Features

| 기능                      | 설명                                                                                                | 세부 설명                                                            |
|-------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| **Data**                | 특수한 포맷(ex. 대규모 데이터 처리를 위한 `parquet`)을 위한 `Dataset` 제공 및 데이터 전처리를 모듈식으로 관리하는 `CollateBase` 클래스 제공  | [[README](zae_engine/data/dataset/README_dataset-ko.md)]         |
| **Loss**                | 대표적인 손실 함수 및 각도, 크기, 영역 기반의 유용한 손실 함수 제공                                                          | [[README](zae_engine/loss/README_loss-ko.md)]                    |
| **Metrics**             | 대표적인 평가 지표 및 집계, 신호, 영역 기반의 유용한 평가 지표 제공                                                          | [[README](zae_engine/metrics/README_metrics-ko.md)]              |
| **Models**              | 다양한 딥러닝 모델의 빌더 대표적인 모델의 구현체 제공 및 모델의 차원 변환기 `dim_converter` 제공                                    | [[README (TBU)](zae_engine/models/builds/README_models-ko.md)]   |
| **NN_night** (Layers)   | 다양한 딥러닝 레이어 및 블록들을 제공                                                                             | [[README (TBU)](zae_engine/nn_night/layers/README_layers-ko.md)] |
| **Operation**           | 알고리즘 기반의 후처리, 변환 등을 제공                                                                            | [[README](zae_engine/operations/README_operation-ko.md)]         |
| **Schedulers**          | 모듈식으로 learning rate을 관리하는 `SchedulerBase` 제공 및 인기 있는 learning rate 스케쥴러 모듈 제공                     | [[README](zae_engine/operations/README_scheduler-ko.md)]         |
| **Trainer**             | 학습 인스턴스를 관리하는 `Trainer` 제공 및 상태 관리, 분산 학습 등 고급 엔지니어링을 지원하는 `add-on` 제공                            | [[README](zae_engine/trainer/README_trainer-ko)]                 |

### Support Frameworks (WIP)
- [x] PyTorch
- [ ] TensorFlow
- [ ] Flax

### zae-command
설치 후 명령창에서 `zae` 명령어를 사용하여 아래의 기능들을 사용할 수 있습니다:
- `zae hello`: 정상 설치 확인
- `zae example`: 예제 스크립트 파일 ***zae_example.py*** 생성
- `zae tree`: 패키지 내 지원하는 클래스 및 함수 프린트


## 설치 방법

### PIP
```bash
pip install zae-engine
# 또는
git+https://github.com/zae-park/zae-engine.git
```

### Poetry
```bash
poetry add zae-engine
# 또는
git+https://github.com/zae-park/zae-engine.git
```


## LICENSE
이 프로젝트는 [Apache License 2.0](./LICENSE)을 따릅니다.
