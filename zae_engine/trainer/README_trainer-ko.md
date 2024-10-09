# Trainer 서브패키지

Trainer 서브패키지는 모델 학습 및 테스트를 관리하고, 다양한 AddOn을 통해 기능을 확장할 수 있도록 하는 핵심적인 기능을 제공하는 서브패키지입니다. 모델 학습 과정에서 반복적인 작업을 단순화하고, 효율적인 관리와 확장이 가능하도록 설계되었습니다.

## 주요 클래스 및 기능

### 1. Trainer 클래스
`Trainer` 클래스는 모델 학습과 테스트를 위한 기본적인 추상 클래스입니다. 이 클래스는 `train_step`과 `test_step`이라는 두 가지 추상 메서드를 가지고 있으며, 이를 통해 각 학습 단계와 테스트 단계의 구체적인 동작을 정의할 수 있습니다.

#### 주요 파라미터
- **model**: 학습할 또는 테스트할 모델(`torch.nn.Module`)
- **device**: 학습에 사용할 장치 (예: 'cuda' 또는 ['cuda:0', 'cuda:1'])
- **mode**: 학습 모드 ('train' 또는 'test')
- **optimizer**: 모델 학습에 사용할 옵티마이저
- **scheduler**: 옵티마이저의 학습률을 조정하기 위한 스케줄러 (옵션)

#### 주요 메서드
- **train_step**: 학습 단계에서 각 배치를 처리하는 추상 메서드. 사용자 정의 학습 로직을 여기에 구현해야 합니다.
- **test_step**: 테스트 단계에서 각 배치를 처리하는 추상 메서드. 사용자 정의 테스트 로직을 여기에 구현해야 합니다.
- **run**: 전체 학습/테스트 과정을 지정된 에폭 수만큼 실행합니다.
- **run_epoch**: 하나의 에폭을 실행합니다.
- **run_batch**: 하나의 배치를 실행합니다.

### 2. AddOn 시스템
Trainer 서브패키지는 **AddOn**을 통해 기능을 확장할 수 있습니다. AddOn은 `Trainer` 클래스에 추가적인 기능을 덧붙이는 방식으로 사용되며, 이를 통해 코드의 재사용성을 높이고 유지보수를 쉽게 합니다.

#### AddOn 예시
- **StateManagerAddon**: 모델, 옵티마이저, 스케줄러의 상태를 저장하고 복원하는 기능을 제공합니다. 학습 중 더 나은 모델을 자동으로 저장할 수 있습니다.
- **WandBLoggerAddon** 및 **NeptuneLoggerAddon**: 학습 중 로깅을 위한 웹 로거 기능을 제공합니다. WandB와 Neptune을 이용하여 실험 결과를 추적할 수 있습니다.
- **MultiGPUAddon**: 다중 GPU에서 분산 학습을 수행할 수 있도록 합니다.

## 사용 예시
Trainer 클래스는 추상 클래스이므로, 이를 상속받아 `train_step`과 `test_step` 메서드를 구현해야 합니다. 예를 들어, 다음과 같이 간단한 Trainer를 정의할 수 있습니다.

```python
from zae_engine.trainer import Trainer

class MyTrainer(Trainer):
    def train_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        return {"loss": loss}

    def test_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        return {"loss": loss}
```

여기서 `train_step`과 `test_step` 메서드는 각각 학습 및 테스트 시의 배치 처리를 담당합니다.

## 확장성 및 모듈성
Trainer 서브패키지는 다양한 AddOn을 통해 확장 가능합니다. AddOn은 다음과 같은 방식으로 추가할 수 있습니다.

```python
from zae_engine.trainer.addons import StateManagerAddon, WandBLoggerAddon

CustomTrainer = Trainer.add_on(StateManagerAddon, WandBLoggerAddon)
trainer = CustomTrainer(model, device, mode='train', optimizer=optimizer, scheduler=scheduler, save_path='./checkpoints', web_logger={"wandb": {"project": "my_project"}})
```

위와 같이 AddOn을 추가하면, 학습 중 상태 저장 및 웹 로깅 기능이 자동으로 포함된 Trainer를 사용할 수 있습니다.

## 결론
Trainer 서브패키지는 모델 학습 과정을 관리하고 효율적으로 확장할 수 있는 강력한 도구를 제공합니다. 기본적인 학습 루프와 테스트 루프를 간단하게 구현할 수 있으며, 다양한 AddOn을 통해 쉽게 기능을 확장하여 복잡한 실험 환경에서도 유연하게 대처할 수 있습니다.

