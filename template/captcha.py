import os
import urllib.request
import time
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from zae_engine.data_pipeline import example_ecg
from zae_engine import trainer, models, measure, data_pipeline
from zae_engine.trainer import mpu_utils

LOOKUP = {k: v for k, v in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")}
LOOKDOWN = {v: k for k, v in LOOKUP.items()}


class CaptchaImgSaver:
    def __init__(self, dst: str = "./"):
        self.dst = dst
        if not os.path.exists(dst):
            os.mkdir(dst)
        self.url = "https://www.ftc.go.kr/captcha.do"

    def run(self, iter: int = 100):
        for i in tqdm(range(iter), desc="Get Captcha images from url "):
            name = str(time.time()).replace(".", "") + ".png"
            urllib.request.urlretrieve(self.url, os.path.join(self.dst, name))


class CaptchaDataset(Dataset):
    def __init__(self, x, y, _type="tuple"):
        self.x = x  # List: path of images
        self.y = y  # List: labels
        self._type = _type
        self.str2emb = lambda label: [LOOKDOWN[l] for l in label]
        self.emb2str = lambda embed: [LOOKUP[e] for e in embed]

    def __len__(self):
        return len(self.x)

    def load_image(self, idx):
        return np.array(Image.open(self.x[idx]))

    def __getitem__(self, idx):
        x = torch.tensor(self.load_image(idx), dtype=torch.float32)[..., :-1]
        x = torch.permute(x, [2, 0, 1])
        y = torch.zeros((36, 5), dtype=torch.float32)
        for i, yy in enumerate(self.str2emb(self.y[idx])):
            y[yy, i] = 1

        if self._type == "tuple":
            return x, y
        elif self._type == "dict":
            return {"x": x, "y": y}
        else:
            raise ValueError


class CaptchaTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(CaptchaTrainer, self).__init__(model, device, mode, optimizer, scheduler)

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
        acc = measure.accuracy(self._to_cpu(y.argmax(1)), self._to_cpu(prediction))
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


class CaptchaModel(models.CNNBase):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        width: int,
        kernel_size: int or tuple,
        depth: int,
        order: int,
        stride: list or tuple,
    ):
        super().__init__(
            ch_in=ch_in, ch_out=ch_out, width=width, kernel_size=kernel_size, depth=depth, order=order, stride=stride
        )
        self.pool = nn.Sequential(nn.Linear(15 * 63, 64), nn.Linear(64, 5))
        self.head = nn.Conv1d(48, 36, kernel_size=1)

    def forward(self, x):
        x = self.body(x)
        flat = x.view(x.shape[0], 48, -1)
        pool = self.pool(flat)
        return self.head(pool)


def core():
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    x_filenames = glob.glob("Z:/dev-zae/chapcha_database/labeled/*.png")
    with open("Z:/dev-zae/chapcha_database/labeled/label.txt", "r") as f:
        y_txt = f.read().split("\n")

    captcha_dataset = CaptchaDataset(x=x_filenames, y=y_txt)
    captcha_loader = DataLoader(dataset=captcha_dataset, batch_size=16)

    captcha_model = CaptchaModel(3, 36, 16, kernel_size=(3, 3), stride=[2, 2], depth=3, order=2)

    captcha_opt = torch.optim.Adam(params=captcha_model.parameters(), lr=1e-5)
    captcha_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=captcha_opt)

    ex_trainer = CaptchaTrainer(
        model=captcha_model, device=device, mode="train", optimizer=captcha_opt, scheduler=captcha_scheduler
    )
    t = time.time()
    ex_trainer.run(n_epoch=100, loader=captcha_loader)
    print(time.time() - t)

    # result = np.concatenate(ex_trainer.inference(loader=ex_loader3), axis=0).argmax(1)
    # print(f"Accuracy: {measure.accuracy(test_y, result):.8f}")
    #
    # return result, ex_loader3.dataset.y


if __name__ == "__main__":
    # cap = CaptchaImgSaver("./outputs")
    # cap.run(iter=10000)

    core()
