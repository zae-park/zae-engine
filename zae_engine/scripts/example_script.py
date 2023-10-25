import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

from zae_engine.data_pipeline import example_ecg
from zae_engine import trainer, models, measure, operation


class ExDataset(data.Dataset):
    def __init__(self, x, y, _type="tuple"):
        self.x = np.array(x, dtype=float)  # [N, dim]
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
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = measure.accuracy(self._to_cpu(y), self._to_cpu(prediction))
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
        acc = measure.accuracy(self._to_cpu(y), self._to_cpu(prediction))
        return {"loss": loss, "output": out, "acc": acc}


if __name__ == "__main__":
    num_data = 50
    batch_size = 8
    num_class = 4
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    ex_x = np.sin(np.linspace(0, 2 * np.pi, 2560))
    ex_y = ex_x > 0.5

    dataset1 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data)
    dataset2 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data, _type="dict")
    ex_loader1 = data.DataLoader(dataset=dataset1, batch_size=batch_size)
    ex_loader2 = data.DataLoader(dataset=dataset2, batch_size=batch_size)

    ex_model = models.Segmentor()

    ex_opt = torch.optim.Adam(params=ex_model.parameters(), lr=1e-2)
    ex_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=ex_opt)

    ex_trainer = ExTrainer(model=ex_model, device=device, mode="train", optimizer=ex_opt, scheduler=ex_scheduler)
    ex_trainer.run(n_epoch=10, loader=ex_loader1, valid_loader=ex_loader2)

    test_y = np.array([ex_y] * num_data)
    test_y[:, 0] = np.arange(num_data) % num_class
    dataset3 = ExDataset(x=[ex_x] * num_data, y=test_y, _type="dict")
    ex_loader3 = data.DataLoader(dataset=dataset3, batch_size=batch_size)

    result = np.concatenate(ex_trainer.inference(loader=ex_loader3), axis=0).argmax(1)
    print(f"Accuracy: {measure.accuracy(test_y, result):.6f}")

    bi_mat = np.zeros((num_class, num_class))
