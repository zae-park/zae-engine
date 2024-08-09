import ssl
import datetime
from collections import namedtuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

from zae_engine import trainer, models, metrics
from zae_engine.models.builds import CNNBase
from zae_engine.nn_night.blocks import BasicBlock
from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.metrics.confusion import confusion_matrix, print_confusion_matrix
from zae_engine.trainer.addons import MultiGPUAddon

ssl._create_default_https_context = ssl._create_unverified_context


# Define the Dataset class
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
            return x, y
        elif self._type == dict:
            return {"x": x, "y": y}
        elif self._type == namedtuple:
            return namedtuple("item", ["x", "y"])(x, y)
        else:
            raise ValueError


# Define the Trainer class with MultiGPU support
class ExTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str = "train", optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(ExTrainer, self).__init__(model, device, mode, optimizer, scheduler, scheduler_step_on_batch=True)

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
    # Initialize distributed training
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(torch.distributed.get_rank())
    else:
        raise RuntimeError("CUDA is not available. Multi-GPU training requires CUDA.")

    kickoff_time = datetime.datetime.now()
    epochs = 2
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device(f"cuda:{torch.distributed.get_rank()}" if torch.cuda.is_available() else "cpu")

    mnist = fetch_openml("mnist_784")
    x, y = mnist.data, mnist.target.astype(int)

    x_train, x, y_train, y = train_test_split(x, y, test_size=0.2, random_state=1111, stratify=y)
    x_valid, x_test, y_valid, y_test = train_test_split(x, y, test_size=0.5, random_state=1111, stratify=y)

    train_loader = DataLoader(dataset=ExDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=ExDataset(x_valid, y_valid), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(dataset=ExDataset(x_test, y_test), batch_size=batch_size * 2, shuffle=False)

    model = CNNBase(block=BasicBlock, ch_in=1, ch_out=10, width=64, groups=1, dilation=1, layers=[2, 2, 2, 2])
    opt = Adam(params=model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingScheduler(optimizer=opt, total_iters=epochs * len(train_loader))

    # Apply MultiGPUAddon
    TrainerWithMultiGPU = trainer.Trainer.add_on(MultiGPUAddon)
    extrainer = TrainerWithMultiGPU(
        model,
        device=device,
        optimizer=opt,
        scheduler=scheduler,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )

    extrainer.run(n_epoch=epochs, loader=train_loader, valid_loader=valid_loader)

    test_result = np.stack(extrainer.inference(loader=test_loader))
    confusion_mat = confusion_matrix(y_test, test_result, num_classes=10)
    print_confusion_matrix(confusion_mat)

    elapsed_time = datetime.datetime.now() - kickoff_time
    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()
