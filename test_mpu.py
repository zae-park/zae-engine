import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from zae_engine.trainer import Trainer  # 사용자 정의 Trainer 클래스를 임포트
from zae_engine.trainer.addons.mpu import MultiGPUAddon  # MultiGPUAddon 임포트


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


# MultiGPUAddon 적용된 Trainer 클래스
class MultiGPUTrainer(MyTrainer):
    def __init__(self, *args, init_method="tcp://localhost:12355", **kwargs):
        self.init_method = init_method
        super().__init__(*args, **kwargs)


def main():
    device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]  # 사용 가능한 모든 GPU
    assert len(device_list) > 1, "이 스크립트는 여러 GPU에서 실행되어야 합니다."

    # 데이터셋 및 데이터로더
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 옵티마이저, 스케줄러 설정
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = None  # 필요한 경우 스케줄러 설정

    # Trainer에 MultiGPUAddon 적용
    MultiGPUTrainerClass = MyTrainer.add_on(MultiGPUAddon)
    
    # Trainer 인스턴스 생성 & 학습
    trainer = MultiGPUTrainerClass(model=model, device=device_list, mode='train', optimizer=optimizer, scheduler=scheduler, init_method="tcp://localhost:12355")
    
    # 기본 train_process 사용하여 MultiGPU 학습 수행
    trainer.run(n_epoch=10, loader=train_loader)

    # 모델 저장
    torch.save(model.state_dict(), 'dummy_model_mgpu.pth')
    print("모델이 'dummy_model_mgpu.pth'로 저장되었습니다.")


if __name__ == "__main__":
    main()
