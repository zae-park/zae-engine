# MultiGPUAddon을 사용한 학습 예제

이 문서에서는 MultiGPUAddon을 사용하여 여러 GPU에서 모델을 학습하는 방법에 대해 설명합니다. 이 예제는 앞서 단일 GPU 학습에서 사용한 Trainer 클래스를 확장하여, 여러 GPU에서 병렬 학습을 수행하는 방법을 보여줍니다.

## Requirement

- Python 3.10 이상
- PyTorch 2.0 이상
- CUDA 호환 GPU 2개 이상

## Script

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torch.distributed as dist
from zae_engine.trainer import Trainer  # 사용자 정의 Trainer 클래스를 임포트
from zae_engine.trainer.addons.mpu import MultiGPUAddon  # MultiGPUAddon 임포트
from zae_engine.schedulers import CosineAnnealingScheduler


# 더미 데이터셋
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()  # 평균값이 0.5보다 크면 1, 작으면 0

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

# 사용 시, 이전과 동일하게 적용하면 됩니다.
def main():
    device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]  # 사용 가능한 모든 GPU
    assert len(device_list) > 1, "이 스크립트는 여러 GPU에서 실행되어야 합니다."

    # 데이터셋 및 데이터로더
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 옵티마이저, 스케줄러 설정
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer, total_iters=100)  # total_iters 조정

    # Trainer 인스턴스 생성 & 학습
    trainer = MultiGPUTrainer(model=model, device=device_list, mode='train', optimizer=optimizer, scheduler=scheduler, init_method="tcp://localhost:12355")
    
    # MultiGPU 학습 수행
    trainer.run(n_epoch=10, loader=train_loader)

    # 모델 저장
    torch.save(model.state_dict(), 'dummy_model_mgpu.pth')
    print("모델이 'dummy_model_mgpu.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()

```
스크립트는 사용 가능한 모든 GPU를 자동으로 감지하여 병렬로 학습을 수행합니다.


### 설명

	•	DummyDataset: [0, 1] 범위의 임의 데이터와 레이블을 생성합니다.
	•	DummyModel: 이진 분류를 위한 간단한 단일 선형 레이어를 가진 신경망을 준비합니다.
	•	MyTrainer: 순전파 및 역전파, loss 계산과 모델 최적화를 포함하는 사용자 정의 트레이너 클래스입니다.
	•	MultiGPUAddon: 이 Addon을 통해 Trainer 클래스는 여러 GPU를 사용하여 병렬 학습을 수행할 수 있습니다.
	•	학습 프로세스: Trainer.run 메소드를 사용하여 epoch 수만큼 모델을 학습시킵니다. 학습 후 모델은 파일(dummy_model_mgpu.pth)로 저장됩니다.

### 참고 사항

	•	MultiGPUAddon을 사용하려면 시스템에 CUDA 호환 GPU가 2개 이상 있어야 합니다.
	•	이 예제는 PyTorch의 DistributedDataParallel을 활용하여 각 GPU에서 데이터를 병렬로 처리하고 모델을 학습시킵니다.
	•	init_method는 네트워크 상에서 프로세스 간 통신을 설정하는 방법을 지정합니다. 이 예제에서는 localhost를 사용하고 있습니다.

**라이센스**

이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다.