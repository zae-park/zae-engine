# Trainer를 사용한 학습 예제

사용자 정의 `Trainer` 클래스를 사용하여 학습을 수행하는 방법에 대한 예제입니다.

## Requirement

- Python 3.10 이상
- PyTorch 2.0 이상
- CUDA 호환 GPU (선택 사항, GPU가 없으면 코드가 CPU에서 실행됩니다)

## Script

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from zae_engine.trainer import Trainer  # 사용자 정의 Trainer 클래스를 임포트


# 더미 데이터셋
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()  # 평균값이 0.5보다 크면 1, 작으면 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 사용자 정의 Trainer 클래스
class MyTrainer(Trainer):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

    def test_step(self, batch):
        return self.train_step(batch)

    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 및 데이터로더
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 옵티마이저, 스케줄러 설정
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = None  # 필요한 경우 스케줄러 설정

    # Trainer 인스턴스 생성 & 학습
    trainer = MyTrainer(model=model, device=device, mode='train', optimizer=optimizer, scheduler=scheduler)
    trainer.run(n_epoch=10, loader=train_loader)

    # 모델 저장
    torch.save(model.state_dict(), 'dummy_model.pth')
    print("모델이 'dummy_model.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()
```

스크립트는 자동으로 GPU가 사용 가능한지 감지하고 학습에 사용할 것입니다. GPU가 없으면 CPU를 사용하여 학습을 진행합니다.

### 설명

	•	DummyDataset: [0, 1] 범위의 임의 데이터와 레이블을 생성합니다.
	•	DummyModel: 이진 분류를 위한 간단한 단일 선형 레이어를 가진 신경망을 준비합니다.
	•	MyTrainer: 순전파 및 역전파, loss 계산과 모델 최적화를 포함하는 사용자 정의 트레이너 클래스입니다.
	•	학습 프로세스: Trainer.run 메소드를 사용하여 epoch 수만큼 모델을 학습시킵니다. 학습 후 모델은 파일(dummy_model.pth)로 저장됩니다.

### 참고 사항

	•	제공된 Trainer 클래스는 추상 클래스이므로, train_step 및 test_step 메소드는 하위 클래스에서 구현해야 합니다. 이 예제에서는 MyTrainer가 이러한 메소드를 구현합니다.
	•	학습 스크립트는 다양한 데이터셋, 모델 및 학습 구성에 맞게 수정하기 쉽도록 설계되었습니다.

**라이센스**

이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다.