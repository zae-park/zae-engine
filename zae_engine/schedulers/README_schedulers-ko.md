# 스케줄러 서브패키지

이 서브패키지는 딥러닝 모델 학습 과정에서 학습률(Learning Rate)을 효과적으로 관리하기 위한 다양한 스케줄러들을 제공합니다. 사용자는 기본 제공되는 스케줄러를 사용하거나, 필요에 따라 직접 커스터마이즈하여 사용할 수 있습니다.

## 주요 클래스

### 1. `SchedulerBase`

`SchedulerBase`는 모든 학습률 스케줄러들이 상속받아야 하는 기본 클래스입니다. 이 클래스는 PyTorch의 `LRScheduler`를 확장하여 추가적인 기능을 제공합니다. 사용자는 이 클래스를 상속받아 원하는 학습률 스케줄러를 구현할 수 있습니다.

#### 주요 메소드
- `get_lr()`: 현재 에포크(epoch)에 대한 학습률을 반환합니다. 이 메소드는 반드시 사용자 정의 클래스에서 구현되어야 합니다.

### 2. `SchedulerChain`

`SchedulerChain`은 여러 스케줄러들을 순차적으로 연결하여 사용할 수 있는 기능을 제공합니다. 예를 들어, 모델 학습 초기에는 워밍업(Warm-Up) 스케줄러를 사용하고 이후에는 코사인 감소 스케줄러를 사용하는 방식으로 다양한 학습률 변화를 조합할 수 있습니다.

#### 주요 속성 및 메소드
- `sanity_check()`: 모든 스케줄러들이 동일한 옵티마이저를 사용하고 있는지 확인합니다.
- `step()`: 스케줄러의 현재 상태에 따라 학습률을 업데이트합니다.
- `get_lr()`: 현재 에포크(epoch)에 대한 학습률을 반환합니다.

### 3. `WarmUpScheduler`

`WarmUpScheduler`는 학습 초기의 모델 학습률을 점진적으로 증가시키는 스케줄러입니다. 초기 학습률이 낮은 경우 안정적인 학습을 할 수 있도록 도와줍니다.

### 4. `CosineAnnealingScheduler`

`CosineAnnealingScheduler`는 코사인 함수 형태로 학습률을 감소시키는 스케줄러입니다. 학습 후반부로 갈수록 학습률이 천천히 줄어들며, 모델이 최적의 파라미터에 점진적으로 수렴하도록 돕습니다.

## 사용 예시

```python
from torch.optim import Adam
from zae_engine.schedulers import WarmUpScheduler, CosineAnnealingScheduler, SchedulerChain

# 옵티마이저 정의
optimizer = Adam(model.parameters(), lr=0.01)

# 스케줄러 정의
warmup_scheduler = WarmUpScheduler(optimizer, total_iters=500, eta_min=0)
cosine_scheduler = CosineAnnealingScheduler(optimizer, total_iters=1000, eta_min=0)

# 스케줄러 체인 사용
scheduler_chain = SchedulerChain(warmup_scheduler, cosine_scheduler)

for epoch in range(num_epochs):
    train(...)
    validate(...)
    scheduler_chain.step()
```

위 코드에서는 `WarmUpScheduler`와 `CosineAnnealingScheduler`를 체인으로 연결하여 사용하는 예시를 보여줍니다. 학습 초기에는 `WarmUpScheduler`가 사용되며, 이후 `CosineAnnealingScheduler`로 학습률이 전환됩니다.

## 결론
스케줄러 서브패키지는 학습 과정에서의 학습률 변화를 세밀하게 제어하고자 할 때 유용합니다. `SchedulerBase` 클래스를 상속받아 직접 커스텀 스케줄러를 구현할 수도 있으며, 기본 제공되는 `WarmUpScheduler`, `CosineAnnealingScheduler`, `SchedulerChain` 등을 사용하여 효율적인 학습률 관리가 가능합니다.

