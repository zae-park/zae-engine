# 단일 GPU를 이용한 더미 데이터 학습 예제

이 저장소는 사용자 정의 `Trainer` 클래스를 사용하여 단일 GPU에서 학습을 수행하는 방법에 대한 예제를 포함하고 있습니다.

## 요구 사항

- Python 3.x
- PyTorch 1.9 이상
- CUDA 호환 GPU (선택 사항, GPU가 없으면 코드가 CPU에서 실행됩니다)

## 예제 구성 요소

- **DummyDataset**: 임의의 데이터를 생성하고 특정 임계값을 기준으로 레이블을 할당하는 간단한 데이터셋입니다.
- **DummyModel**: 이진 분류를 위한 기본 선형 모델입니다.
- **MyTrainer**: 학습 및 평가 단계를 처리하는 사용자 정의 트레이너 클래스입니다.
- **single_gpu_training.py**: 단일 GPU에서 학습 프로세스를 설정하는 스크립트입니다.

## `single_gpu_training.py` 스크립트

아래는 `single_gpu_training.py`의 코드 예시입니다:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer  # 사용자 정의 Trainer 클래스를 임포트

# 더미 데이터셋
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()  # 평균값이 0.5보다 크면 1, 작으면 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 간단한 선형 모델
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 2)  # 입력 10차원, 출력 2차원

    def forward(self, x):
        return self.linear(x)

# 사용자 정의 Trainer 클래스
class MyTrainer(Trainer):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

    def test_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

def main():
    # 데이터셋 및 데이터로더
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 옵티마이저, 스케줄러 설정
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = None  # 필요한 경우 스케줄러 설정

    # Trainer 인스턴스 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = MyTrainer(model=model, device=device, mode='train', optimizer=optimizer, scheduler=scheduler)

    # 학습 수행
    trainer.run(n_epoch=10, loader=train_loader)

    # 모델 저장
    torch.save(model.state_dict(), 'dummy_model.pth')
    print("모델이 'dummy_model.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()
```

실행 방법

	1.	이 저장소를 클론하고 프로젝트 디렉토리로 이동합니다.
	2.	필요한 종속성이 설치되었는지 확인합니다.
	3.	학습 스크립트를 실행합니다:

python single_gpu_training.py

스크립트는 자동으로 GPU가 사용 가능한지 감지하고 학습에 사용할 것입니다. GPU가 없으면 CPU를 사용하여 학습을 진행합니다.

설명

	•	DummyDataset: [0, 1] 범위의 임의 데이터를 생성하고 데이터 포인트가 특정 범위에 속하는지 여부에 따라 레이블을 할당합니다.
	•	DummyModel: 이진 분류를 위한 간단한 단일 선형 레이어를 가진 신경망입니다.
	•	MyTrainer: 포워드 및 백워드 패스, 손실 계산 및 모델 최적화를 포함한 학습 루프를 처리하는 사용자 정의 트레이너 클래스입니다.
	•	학습 프로세스: Trainer.run 메소드를 사용하여 지정된 에포크 수만큼 모델을 학습시킵니다. 학습 후 모델은 파일(dummy_model.pth)로 저장됩니다.

참고 사항

	•	제공된 Trainer 클래스는 추상 클래스이므로, train_step 및 test_step 메소드는 하위 클래스에서 구현해야 합니다. 이 예제에서는 MyTrainer가 이러한 메소드를 구현합니다.
	•	single_gpu_training.py 스크립트는 다양한 데이터셋, 모델 및 학습 구성에 맞게 수정하기 쉽도록 설계되었습니다.

라이센스

이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다.