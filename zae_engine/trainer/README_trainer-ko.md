# Trainer 서브패키지 개요

**zae-engine** 라이브러리의 Trainer 서브패키지는 딥러닝 모델 학습을 관리하고 다양한 애드온을 통해 기능을 확장할 수 있는 강력한 도구를 제공합니다. `Trainer` 클래스는 학습 프로세스를 감독하는 핵심 구성 요소로, 사용자 정의 학습 및 테스트 단계를 구현할 수 있는 추상 메서드를 제공합니다. 애드온 시스템을 통해 사용자는 학습 워크플로우를 쉽게 확장하고 사용자화할 수 있습니다.

## 주요 클래스 및 기능

### Trainer

`Trainer`는 모델의 학습 루틴을 관리하기 위해 설계된 추상 기본 클래스입니다. 사용자는 이를 상속받아 학습(`train_step`)과 테스트(`test_step`) 중 수행할 동작을 정의해야 합니다.

#### 주요 속성
- **model**: 학습할 모델(`torch.nn.Module`).
- **device**: 모델 학습을 위한 디바이스(예: 'cuda' 또는 ['cuda:0', 'cuda:1']).
- **optimizer**: 학습에 사용되는 옵티마이저(`torch.optim.Optimizer`).
- **scheduler**: 학습률 스케줄러(`torch.optim.lr_scheduler._LRScheduler`).
- **log_bar**: 학습 중 진행 바를 표시할지 여부.
- **gradient_clip**: 그래디언트 클리핑 값(기본값: 0.0).
- **log_train & log_test**: 각 epoch 동안 학습 및 검증 데이터를 기록하는 사전 형태의 로그.

#### 주요 기능
- **디바이스 관리**: 모델과 데이터를 적절한 디바이스에 할당하여 단일 GPU 및 다중 GPU 학습을 지원합니다.
- **배치 처리**: 학습 또는 테스트 모드에서 각 배치를 처리하며, 학습 모드에서는 역전파도 수행합니다.
- **로그 관리**: 각 epoch 또는 배치에서 손실 및 기타 메트릭 정보를 수집하고, 로그를 저장하거나 출력할 수 있습니다.
- **그래디언트 클리핑**: 학습 중 그래디언트 폭발을 방지하기 위해 그래디언트를 선택적으로 클리핑할 수 있습니다.
- **스케줄러 통합**: 학습률 스케줄러를 epoch 또는 배치 수준에서 적용하여 학습 과정을 적응적으로 제어합니다.
- **상태 저장 및 불러오기**: 모델 가중치를 저장하고 저장된 가중치를 불러와 추론 또는 학습을 이어갈 수 있습니다.

#### 주요 메서드
- **train_step(batch)**: 각 학습 단계에서 수행할 동작 정의.
- **test_step(batch)**: 각 테스트 단계에서 수행할 동작 정의.
- **run(n_epoch, loader, valid_loader)**: 지정된 epoch 수만큼 학습 또는 테스트 실행.
- **run_epoch(loader)**: 단일 epoch 동안 학습 또는 테스트 실행.
- **run_batch(batch)**: 단일 배치에 대한 학습 또는 테스트 실행.
- **add_on(*add_on_cls)**: 애드온을 통해 `Trainer` 클래스의 기능 확장.
- **metric_on_epoch_end()**: 각 epoch 종료 시 사용자 정의 메트릭을 정의하고 `log_train` 및 `log_test`를 활용할 수 있음.

### ProgressChecker

`ProgressChecker`는 학습 또는 테스트 진행 상황을 추적하는 도우미 클래스입니다. epoch 및 단계 수를 관리하여 사용자가 학습 상태를 쉽게 모니터링할 수 있도록 합니다.

---

## 애드온 기능

Trainer 서브패키지는 `AddOnBase` 클래스를 통해 분산 학습, 상태 관리 및 웹 로깅과 같은 기능을 확장할 수 있습니다.

### 주요 애드온

#### StateManagerAddon
모델 상태, 옵티마이저 상태 및 스케줄러 상태를 저장하고 불러오는 기능을 제공합니다. `.ckpt` 및 `.safetensor` 형식을 지원하여 보안성과 유연성을 제공합니다.

#### MultiGPUAddon
DDP(Distributed Data Parallel)를 사용하여 여러 GPU에서 분산 학습을 지원합니다. PyTorch의 분산 학습 유틸리티와 통합되어 대규모 모델의 학습 속도를 크게 향상시킬 수 있습니다.

#### WandBLoggerAddon / NeptuneLoggerAddon
Weights & Biases(WandB) 또는 Neptune과 같은 외부 서비스를 사용하여 학습 프로세스를 실시간으로 모니터링할 수 있습니다. 학습 메트릭을 자동으로 로깅하며, 사용자는 원격으로 진행 상황을 추적할 수 있습니다.

---

## 사용 예제

### 기본 사용법

```python
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import StateManagerAddon, WandBLoggerAddon

# StateManager 및 WandBLogger 애드온 추가
MyTrainer = Trainer.add_on(StateManagerAddon, WandBLoggerAddon)

trainer = MyTrainer(
    model=my_model,
    device='cuda',
    mode='train',
    optimizer=my_optimizer,
    scheduler=my_scheduler,
    save_path='./checkpoints',
    web_logger={"wandb": {"project": "my_project"}},
)

trainer.run(n_epoch=10, loader=train_loader, valid_loader=valid_loader)
```

### 고급 기능: MultiGPUAddon 사용

`MultiGPUAddon`은 여러 GPU에서 모델을 병렬로 학습시킬 수 있는 강력한 도구입니다. 이 애드온은 PyTorch의 DDP(Distributed Data Parallel)를 활용하여 모든 GPU에서 모델과 데이터를 올바르게 동기화합니다.

#### MultiGPUAddon을 활용한 학습 예제

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons.mpu import MultiGPUAddon
from zae_engine.schedulers import CosineAnnealingScheduler

# 더미 데이터셋
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# MultiGPUTrainer 정의
class MultiGPUTrainer(Trainer.add_on(MultiGPUAddon)):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

    def test_step(self, batch):
        return self.train_step(batch)

# 메인 함수
def main():
    device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    assert len(device_list) > 1, "이 스크립트는 여러 GPU에서 실행되어야 합니다."

    # 데이터셋 및 데이터로더
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 옵티마이저 및 스케줄러 설정
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer, total_iters=100)

    # Trainer 인스턴스 생성 및 학습 실행
    trainer = MultiGPUTrainer(
        model=model,
        device=device_list,
        mode='train',
        optimizer=optimizer,
        scheduler=scheduler,
        init_method="tcp://localhost:12355"
    )
    
    trainer.run(n_epoch=10, loader=train_loader)
    torch.save(model.state_dict(), 'dummy_model_mgpu.pth')
    print("모델이 'dummy_model_mgpu.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()
```

---

### FAQ 및 일반적인 문제

- **Q: `init_method`에서 'Address already in use' 오류가 발생하면 어떻게 하나요?**
  **A**: `init_method`에 지정된 포트(예: `12355`)가 이미 사용 중인지 확인하세요. 포트 번호를 변경하거나 `os.environ["MASTER_PORT"]`를 사용하여 동적으로 설정하세요.

- **Q: 여러 GPU를 사용할 때 학습 속도가 느려지는 이유는 무엇인가요?**
  **A**: 배치 크기가 충분히 큰지 확인하세요. DDP는 작업이 GPU 간에 균등하게 분배될 때 가장 잘 작동합니다. 데이터 로딩이 최적화되었는지도 확인하세요.

- **Q: 사용자 정의 메트릭을 추가하려면 어떻게 해야 하나요?**
  **A**: Trainer 서브클래스에서 `metric_on_epoch_end()` 메서드를 재정의하고 사용자 정의 메트릭 딕셔너리를 반환하세요.

---

