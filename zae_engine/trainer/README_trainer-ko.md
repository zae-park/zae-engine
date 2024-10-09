# Trainer 서브패키지 개요

**zae-engine** 라이브러리의 Trainer 서브패키지는 딥러닝 모델 학습을 위한 유연하고 강력한 프레임워크를 제공합니다. 이 서브패키지의 핵심 구성 요소는 `Trainer` 클래스이며, 사용자가 맞춤형 학습 프로세스를 구현할 수 있도록 도와줍니다. 또한 다양한 애드온을 통해 기능을 확장하여 학습 프로세스를 더욱 향상시킬 수 있습니다. 이 README는 Trainer 서브패키지의 주요 기능과 사용 가능한 애드온을 설명합니다.

### 주요 구성 요소

- **Trainer**: `Trainer`는 학습과 테스트 프로세스를 관리하는 추상 클래스입니다. 사용자는 `Trainer`를 상속받아 `train_step()`과 `test_step()` 메서드를 구현해야 합니다. 이 메서드들은 각각 학습과 테스트 시의 동작을 정의합니다.

  주요 기능:
  - **디바이스 관리**: 모델과 데이터를 적절한 디바이스에 할당하여 단일 GPU 및 다중 GPU 학습을 지원합니다.
  - **배치 처리**: 학습 또는 테스트 모드에서 각 배치를 처리하며, 학습 모드에서는 역전파도 수행합니다.
  - **로그 관리**: 각 epoch 또는 배치에서 손실 및 기타 메트릭 정보를 수집하고, 로그를 저장하거나 출력할 수 있습니다.
  - **그래디언트 클리핑**: 학습 중 그래디언트 폭발을 방지하기 위해 그래디언트를 선택적으로 클리핑할 수 있습니다.
  - **스케줄러 통합**: 학습률 스케줄러를 epoch 또는 배치 수준에서 적용하여 학습 과정을 적응적으로 제어합니다.
  - **상태 저장 및 불러오기**: 모델 가중치를 저장하고 저장된 가중치를 불러와 추론 또는 학습을 이어갈 수 있습니다.

- **애드온 시스템**: `Trainer` 클래스는 애드온을 통해 추가 기능을 확장할 수 있습니다. `Trainer.add_on()` 메서드를 사용하여 원하는 애드온 클래스를 전달하면 기능을 추가할 수 있습니다.

### 사용 가능한 애드온

1. **StateManagerAddon**
   - 모델, 옵티마이저, 스케줄러의 상태를 저장하고 불러오는 기능을 제공합니다. `.ckpt` 또는 `.safetensor` 형식으로 상태를 저장할 수 있어 저장과 보안에서 유연성을 제공합니다.

2. **WandBLoggerAddon & NeptuneLoggerAddon**
   - Weights and Biases (WandB) 또는 Neptune과 같은 외부 서비스에 학습 메트릭을 기록할 수 있습니다. 사용자는 학습 진행 상황을 원격으로 모니터링하고, 메트릭을 추적하며, 모델 성능을 시각화할 수 있습니다.

3. **MultiGPUAddon**
   - PyTorch의 분산 학습 유틸리티를 사용하여 여러 GPU에서 모델을 학습할 수 있도록 지원합니다. 각 GPU에 대한 프로세스를 생성하고 `DistributedDataParallel (DDP)`를 사용해 모델 업데이트를 동기화합니다. 대규모 모델의 학습 시간을 단축하는 데 유용합니다.

### Trainer 사용 방법

1. **커스텀 Trainer 생성**: `Trainer`를 상속받아 `train_step()`과 `test_step()` 메서드를 구현합니다. 각 메서드는 학습과 테스트 중 각 배치에서 수행할 동작을 정의합니다.
   
   ```python
   from zae_engine.trainer import Trainer

   class MyTrainer(Trainer):
       def train_step(self, batch):
           x, y = batch['input'], batch['target']
           outputs = self.model(x)
           loss = self.criterion(outputs, y)
           return {'loss': loss}

       def test_step(self, batch):
           x, y = batch['input'], batch['target']
           outputs = self.model(x)
           loss = self.criterion(outputs, y)
           return {'loss': loss}
   ```

2. **애드온 통합**: Trainer를 확장해야 할 경우, `add_on()` 메서드를 사용합니다.
   
   ```python
   from zae_engine.trainer.addons import StateManagerAddon, WandBLoggerAddon

   CustomTrainer = MyTrainer.add_on(StateManagerAddon, WandBLoggerAddon)
   trainer = CustomTrainer(
       model=my_model,
       device='cuda',
       mode='train',
       optimizer=my_optimizer,
       scheduler=my_scheduler,
       save_path='./model_states'
   )
   ```

3. **학습 실행**: `trainer.run()`을 호출하여 학습을 시작합니다.

   ```python
   trainer.run(n_epoch=50, loader=train_loader, valid_loader=valid_loader)
   ```

### 요약

Trainer 서브패키지는 머신 러닝 모델 학습을 위한 적응적이고 확장 가능한 기반을 제공합니다. 추상 메서드와 애드온을 통해 사용자에게 학습 프로세스를 제어할 수 있는 유연성을 제공하며, 다중 GPU 지원 및 외부 로깅 서비스 통합과 같은 다양한 내장 기능을 제공합니다.

