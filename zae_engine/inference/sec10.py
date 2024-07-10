from functools import wraps
from typing import Union, Dict, Optional, List, Callable, Tuple

import numpy as np
import torch
from scipy import signal as sig
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from zae_engine.trainer import Trainer

from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.models.builds.cnn import CNNBase
from zae_engine.models.converter import dim_converter
from zae_engine.nn_night.blocks import BasicBlock


def filter_signal(x: np.ndarray, sample_rate: int, btype: Optional[str] = "bandpass"):
    """filter ecg signal"""
    nyq = sample_rate * 0.5
    if btype == "bandpass":
        x = sig.filtfilt(*sig.butter(2, [0.5 / nyq, 50 / nyq], btype=btype), x)
    if btype == "bandstop":
        x = sig.filtfilt(*sig.butter(2, [59.9 / nyq, 60.1 / nyq], btype=btype), x)
    return x


def scale_signal(x: np.ndarray):
    scaler = MinMaxScaler()
    batch = []
    for subset_x in x:
        scaler.fit(np.expand_dims(subset_x, 1))
        subset_x = scaler.transform(np.expand_dims(subset_x, 1)).squeeze()
        batch.append(subset_x)
    x = np.array(batch)
    return x


class Sec10Trainer(Trainer):
    def __init__(self, model, device, mode, optimizer, scheduler):
        super(Sec10Trainer, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 32

    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        pass

    def test_step(self, batch: dict):
        x = batch["x"]
        mini_x = x.split(self.mini_batch_size, dim=0)
        out = torch.cat([self.model(m_x).argmax(-1).detach().cpu() for m_x in mini_x])
        return {"loss": 0, "output": out}


class Sec10Collate:
    def __init__(
        self,
        sequence: Optional[tuple] = ("chunk", "filtering", "scaling"),
        n_cls: Optional[int] = 7,
        cutoff: Optional[List[float]] = (0.5, 50),
        sampling_rate: Optional[int] = 250,
        hot: Optional[bool] = True,
    ):
        super().__init__()
        self.sequence = sequence
        self.n_cls = n_cls
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.is_hot = hot
        self.n_cls = n_cls

    def chunk(self, batch: dict) -> Dict:
        x = batch["x"]
        modulo = x.shape[-1] % 2500
        if modulo != 0:
            x = x[:, :-modulo]
        batch["x"] = x.reshape(-1, self.sampling_rate * 10)
        return batch

    def hot(self, batch: dict) -> Dict:
        onehot_table = np.eye(self.n_cls)
        batch["y"] = onehot_table[[batch["y"]]]
        return batch

    def filtering(self, batch: Dict, bandpass_only: Optional[bool] = False) -> Dict:
        x = filter_signal(batch["x"], self.sampling_rate)

        if bandpass_only:
            batch["x"] = np.array(x)
            return batch

        x = filter_signal(x, self.sampling_rate, btype="bandstop")
        batch["x"] = np.array(x)
        return batch

    @staticmethod
    def scaling(batch: Dict) -> Dict:
        batch["x"] = scale_signal(batch["x"])
        return batch

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        batches = {"x": [], "y": []}
        for b in batch:
            for seq in self.sequence:
                if seq == "chunk":
                    b = self.chunk(b)
                elif seq == "filtering":
                    b = self.filtering(b)
                elif seq == "scaling":
                    b = self.scaling(b)
                else:
                    pass
            if len(b.keys()) >= 2:
                if self.is_hot:
                    batches["y"].append(self.hot(b)["y"])
                else:
                    batches["y"].append(b["y"])
            batches["x"].append(b["x"])
        batches["x"] = np.concatenate(batches["x"])
        batches["x"] = torch.from_numpy(batches["x"]).float().unsqueeze(1)
        batches["y"] = np.concatenate(batches["y"]) if batches["y"] else []
        return batches

    def wrap(self, func: Callable = None):
        if func is None:
            func = self.__call__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func


class Sec10Dataset(Dataset):
    def __init__(self, ecg_set: np.ndarray, label_set: Optional[Union[List, Tuple]] = None):
        self.ecg_set = ecg_set.reshape(1, -1) if len(ecg_set.shape) == 1 else ecg_set
        self.label_set = label_set

    def __len__(self):
        return len(self.ecg_set)

    def __getitem__(self, idx):
        ecg = self.ecg_set[idx].reshape(1, -1)
        if self.label_set is not None:
            label = self.label_set[idx]
            return {"x": ecg, "y": label}
        else:
            return {"x": ecg}


def core(x: np.ndarray, batch_size: int):
    assert len(x.shape) < 3, f"Expect less than 3-D array, but receive {len(x.shape)}-D array."
    collate = Sec10Collate()

    dataset = Sec10Dataset(x)
    ex_loader1 = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate.wrap(), shuffle=False)

    model = CNNBase(BasicBlock, 1, 9, 16, [2, 2, 2, 2])
    cvtr = dim_converter.DimConverter(model)
    model = cvtr.convert("2d -> 1d")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = CosineAnnealingScheduler(optimizer, total_iters=100)
    trainer = Sec10Trainer(model, device, "test", optimizer, scheduler)
    trainer.inference(ex_loader1)

    result = np.concatenate(trainer.log_test["output"])
    result = result.reshape(len(dataset), -1)
    return result


if __name__ == "__main__":
    sample = np.zeros((1, 2500))
    core(sample, batch_size=10)
