<div>
<p align="center">
  <img src="assets/img/spinning_ascii_donut.gif" style="border-radius: 20px">
  <br />
</p>

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
</div>

# zae-engine
______________________________________________________________________


**Accelerate AI project !**

`zae-engine`은 다양한 딥러닝 프레임워크들을 정리한 모듈식 프레임워크입니다. AI 프로젝트의 모든 단계를 표준화하고 반복적인 코드를 줄여 data science를 가속하는 것을 목표로 합니다. 또, 코드의 단순성과 유연성을 높여 복잡한 딥러닝 프로젝트를 쉽게 관리하고 유지 보수할 수 있습니다.


## Advantages over unstructured PyTorch
- 빈번하게 사용되는 boilerplate를 정리하여 코드의 가독성을 높입니다.
- 까다로운 엔지니어링을 추상화하여 `zae-engine`이 처리하므로 실수를 줄일 수 있습니다.
- 고급 엔지니어링을 최소한의 코드 수정으로 지원하여 science와 engineering을 디커플링합니다.
- 다양한 구현체들을 일반화하여 유연성을 높입니다.
- 모든 유연성을 유지하면서(`LightningModules`가 여전히 PyTorch 모듈인 것처럼), 반복적인 보일러플레이트 코드를 제거합니다.
- CPU에서 GPU(Apple Silicon, CUDA 등), TPU, 멀티 GPU 또는 멀티 노드 학습으로 손쉽게 전환할 수 있습니다.
- 체크포인트 저장, 로깅 등의 확장을 간단하게 제공하여 손쉽게 커스터마이즈가 가능합니다.
- Dozens of 매력있고 인기있는 model, layer, tool들이 통합되어 있습니다.

## Features

| 기능               | 설명                                                                                                                                                 | 세부 설명                                                                                                                                                                                            |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data             | 특수한 format(ex. 대규모 data 처리를 위한 parquet)을 위한 Dataset 제공<br/> Data preprocessing을 모듈식으로 관리하는 `CollateBase` 클래스 제공                                    | [[Dataset](zae_engine/data/dataset/README_dataset-ko.md)] (TBU)<br/>[[Collate](zae_engine/data/dataset/README_collate-ko.md)] (TBU)                                                                                    |
| Loss             | 대표적인 손실 함수 및 각도, 크기, 영역 별 인기있는 손실함수 제공                                                                                                             | [[Basic](zae_engine/loss/README_basic-ko.md)] (TBU)<br/>[[Angular](zae_engine/loss/README_angular-ko.md)] (TBU)<br/>[[IoU](zae_engine/loss/README_iou-ko.md)] (TBU)<br/>[[Norm](zae_engine/loss/README_norm-ko.md)] (TBU)                    |
| Metrics          | 대표적인 평가 지표 및 집계, 신호, 영역 별 인기있는 평가 지표 제공                                                                                                            | [[Confusion](zae_engine/metrics/README_confusion-ko.md)] (TBU)<br/>[[Count](zae_engine/metrics/README_count-ko.md)] (TBU)<br/>[[IoU](zae_engine/metrics/README_iou-ko.md)] (TBU)<br/>[[Signal](zae_engine/loss/README_signal-ko.md)] (TBU)   |
| Models           | 다양한 딥러닝 모델의 빌더 대표적인 모델의 구현체 제공<br/> 모델의 차원 변환기 `dim_converter` 제공                                                                                  | [[Builder](zae_engine/models/builds/README_builds-ko.md)] (TBU)<br/>[[Foundations](zae_engine/models/foundations/README_foundations-ko.md)] (TBU)<br/>[[Dim Converter](zae_engine/models/converter/README_converter-ko.md)] (TBU) |
| NN_night (Layers) | 다양한 딥러닝 레이어 및 블록들을 제공                                                                                                                              | [[Layers](zae_engine/nn_night/layers/README_layers-ko.md)] (TBU)<br/>[[Blocks](zae_engine/nn_night/blocks/README_blocks-ko.md)] (TBU)                                                                                  |
| Operation        | 알고리즘 기반의 후처리, 변환 등을 제공                                                                                                                             | [[Operations](zae_engine/operations/README_ops-ko.md)] (TBU)                                                                                                                                                |
| Schedulers       | 모듈식으로 learning rate을 관리하는 `SchedulerBase` 제공<br/>인기 있는 learning rate 스케쥴러 모듈 제공                                                                    | [[Scheduler](zae_engine/operations/README_core-ko.md)] (TBU)<br/>[[Modules](zae_engine/operations/README_scheduler-ko.md)] (TBU)                                                                                       |
| Trainer | 학습 인스턴스를 관리하는 `Trainer` 제공<br/> 상태 관리, 분산 학습 등 고급 엔지니어링을 지원하는 add-on 제공 | [[Trainer](zae_engine/trainer/README_trainer-ko.md)]<br/>[[Add On](zae_engine/trainer/README_mpu-ko.md)]                                                                                                    |

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
