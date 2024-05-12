import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from zae_engine.utils import example_ecg
from zae_engine import trainer, models, metrics
from zae_engine.trainer import mpu_utils


num_data = 10000
num_batch = 64
num_class = 4
epoch = 10


class ExDataset(data.Dataset):
    def __init__(self, x, y, _type="tuple"):
        self.x = 200 * np.array(x, dtype=np.float32)  # [N, dim]
        self.y = np.array(y)  # [N, dim]
        self._type = _type

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self._type == "tuple":
            return torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(
                self.y[idx], dtype=torch.long
            )
        elif self._type == "dict":
            return {
                "x": torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0),
                "y": torch.tensor(self.y[idx], dtype=torch.long),
            }
        else:
            raise ValueError


class ExTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(ExTrainer, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x).softmax(1)
        if isinstance(self.model, nn.DataParallel):
            out = self._to_device(out)
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = metrics.accuracy(self._to_cpu(y), self._to_cpu(prediction))
        return {"loss": loss, "output": out, "acc": acc}

    def test_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x).softmax(1)
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = metrics.accuracy(self._to_cpu(y), self._to_cpu(prediction))
        return {"loss": loss, "output": out, "acc": acc}


class ExTrainer2(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(ExTrainer2, self).__init__(model, device, mode, optimizer, scheduler)
        self.mpu_loss = mpu_utils.DataParallelCriterion(torch.nn.CrossEntropyLoss(), device_ids=[0, 1])

    def train_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x)
        prediction = torch.cat(self._to_cpu(*out), dim=0)
        loss = self.mpu_loss(out, y)
        acc = metrics.accuracy(self._to_cpu(y), torch.cat(self._to_cpu(*out), dim=0).argmax(1))
        return {"loss": loss, "output": prediction, "acc": acc}

    def test_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x)
        prediction = torch.cat(self._to_cpu(*out), dim=0)
        loss = self.mpu_loss(out, y)
        acc = metrics.accuracy(self._to_cpu(y), torch.cat(self._to_cpu(*out), dim=0).argmax(1))
        return {"loss": loss, "output": prediction, "acc": acc}


def core0():
    # MPU without DataParallel
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    ex_x, ex_y = example_ecg()
    dataset1 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data)
    ex_loader1 = data.DataLoader(dataset=dataset1, batch_size=num_batch)

    ex_model = models.beat_segmentation(pretrained=True)
    ex_opt = torch.optim.Adam(params=ex_model.parameters(), lr=1e-2)
    ex_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=ex_opt)

    ex_trainer = ExTrainer(model=ex_model, device=device, mode="train", optimizer=ex_opt, scheduler=ex_scheduler)
    t = time.time()
    ex_trainer.run(n_epoch=epoch, loader=ex_loader1)
    print(time.time() - t)

    test_y = np.array([ex_y] * num_data)
    test_y[:, 0] = np.arange(num_data) % num_class
    dataset3 = ExDataset(x=[ex_x] * num_data, y=test_y, _type="dict")
    ex_loader3 = data.DataLoader(dataset=dataset3, batch_size=num_batch)

    result = np.concatenate(ex_trainer.inference(loader=ex_loader3), axis=0).argmax(1)
    print(f"Accuracy: {metrics.accuracy(test_y, result):.8f}")

    return result, ex_loader3.dataset.y


def core1():
    # MPU with DataParallel
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    ex_x, ex_y = example_ecg()
    dataset1 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data)
    ex_loader1 = data.DataLoader(dataset=dataset1, batch_size=num_batch)

    ex_model = models.beat_segmentation(pretrained=True)
    ex_model = nn.DataParallel(
        ex_model, output_device=device.index
    )  # Use MPU & gathering data_pipeline to specific device
    ex_opt = torch.optim.Adam(params=ex_model.parameters(), lr=1e-2)
    ex_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=ex_opt)

    ex_trainer = ExTrainer(model=ex_model, device=device, mode="train", optimizer=ex_opt, scheduler=ex_scheduler)
    ex_trainer.run(n_epoch=epoch, loader=ex_loader1)

    test_y = np.array([ex_y] * num_data)
    test_y[:, 0] = np.arange(num_data) % num_class
    dataset3 = ExDataset(x=[ex_x] * num_data, y=test_y, _type="dict")
    ex_loader3 = data.DataLoader(dataset=dataset3, batch_size=num_batch)

    result = np.concatenate(ex_trainer.inference(loader=ex_loader3), axis=0).argmax(1)
    print(f"Accuracy: {metrics.accuracy(test_y, result):.8f}")

    return result, ex_loader3.dataset.y


def core2():
    # MPU with CustomParallel - Modelparallel & Dataparallel
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    ex_x, ex_y = example_ecg()
    dataset1 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data)
    ex_loader1 = data.DataLoader(dataset=dataset1, batch_size=num_batch)

    ex_model = models.beat_segmentation(pretrained=True)
    ex_model = mpu_utils.DataParallelModel(ex_model)
    ex_opt = torch.optim.Adam(params=ex_model.parameters(), lr=1e-2)
    ex_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=ex_opt)

    ex_trainer = ExTrainer2(model=ex_model, device=device, mode="train", optimizer=ex_opt, scheduler=ex_scheduler)
    ex_trainer.run(n_epoch=epoch, loader=ex_loader1)

    test_y = np.array([ex_y] * num_data)
    test_y[:, 0] = np.arange(num_data) % num_class
    dataset3 = ExDataset(x=[ex_x] * num_data, y=test_y, _type="dict")
    ex_loader3 = data.DataLoader(dataset=dataset3, batch_size=num_batch)

    result = np.concatenate(ex_trainer.inference(loader=ex_loader3), axis=0).argmax(1)
    print(f"Accuracy: {metrics.accuracy(test_y, result):.8f}")

    return result, ex_loader3.dataset.y


if __name__ == "__main__":
    # unittest.main()
    t = time.time

    num_data = 5000
    num_batch = 128
    num_class = 4
    epoch = 5

    t0 = t()
    res0 = core0()
    t1 = t()
    num_batch *= 2
    res1 = core1()
    t2 = t()
    res2 = core2()
    t3 = t()

    print(t1 - t0)
    print(t2 - t1)
    print(t3 - t2)
