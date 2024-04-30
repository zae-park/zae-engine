import os
import datetime
from collections import namedtuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from zae_engine import trainer, models, metrics, data_pipeline
from zae_engine.operation import draw_confusion_matrix, print_confusion_matrix


class ExDataset(Dataset):
    def __init__(self, x, y, _type: type = tuple):
        self.x = torch.tensor(x.to_numpy().reshape(-1, 28, 28), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)
        self.eye = np.eye(len(y.unique()))
        self._type = _type

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        y = self.eye[y.int()]
        if self._type == tuple:
            return (x, y)
        elif self._type == dict:
            return {"x": x, "y": y}
        elif self._type == namedtuple:
            return namedtuple("item", ["x", "y"])(x, y)
        else:
            raise ValueError


class ExModel(models.CNNBase):
    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 10,
        width: int = 2,
        kernel_size: int or tuple = (3, 3),
        depth: int = 3,
        order: int = 1,
        stride: list or tuple = (2, 2),
    ):
        super().__init__(
            ch_in=ch_in, ch_out=ch_out, width=width, kernel_size=kernel_size, depth=depth, order=order, stride=stride
        )
        self.pool = nn.Sequential(nn.Flatten(), nn.AdaptiveAvgPool1d(32))
        self.head = nn.Linear(32, ch_out)

    def forward(self, x):
        out = self.body(x)
        out = self.pool(out)
        return self.head(out)


class ExTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str = "train", optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(ExTrainer, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch):
        x, y = batch
        proba = self.model(x).softmax(1)
        predict = proba.argmax(1)
        loss = F.cross_entropy(proba, y)

        acc = metrics.accuracy(self._to_cpu(y.argmax(1)), self._to_cpu(predict))
        return {"loss": loss, "output": predict, "acc": acc}

    def test_step(self, batch):
        return self.train_step(batch)


def main():
    kickoff_time = datetime.datetime.now()
    epochs = 2
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    mnist = fetch_openml("mnist_784")
    x, y = mnist.data, mnist.target.astype(int)

    x_train, x, y_train, y = train_test_split(x, y, test_size=0.2, random_state=1111, stratify=y)
    x_valid, x_test, y_valid, y_test = train_test_split(x, y, test_size=0.5, random_state=1111, stratify=y)

    train_loader = DataLoader(dataset=ExDataset(x_train, y_train), batch_size=batch_size)
    valid_loader = DataLoader(dataset=ExDataset(x_valid, y_valid), batch_size=batch_size * 2)
    test_loader = DataLoader(dataset=ExDataset(x_test, y_test), batch_size=batch_size * 2)

    model = ExModel()
    opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda x: 0.9**epochs)

    trainer = ExTrainer(model, device=device, optimizer=opt, scheduler=scheduler)

    trainer.run(n_epoch=epochs, loader=train_loader, valid_loader=valid_loader)

    test_result = np.stack(trainer.inference(loader=test_loader))
    confusion_mat = draw_confusion_matrix(y_test, test_result, num_classes=10)
    print_confusion_matrix(confusion_mat)

    elapsed_time = datetime.datetime.now() - kickoff_time


if __name__ == "__main__":
    main()
