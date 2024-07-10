from abc import ABC, abstractmethod

import numpy as np
from scipy import signal
import torch
from torch.nn import functional as F
from einops import repeat, reduce

from zae_engine.operation import label_to_onoff


class Chunker:
    """
    Class for chunking the data in the batch.

    Parameters
    ----------
    n : int
        Number of chunks.
    th : float
        Threshold for binarization.

    Methods
    -------
    __call__(batch: Tuple[torch.Tensor, torch.Tensor, Any]) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        Chunk the data in the batch.
    """

    def __init__(self, n: int, th: float):
        self.n = n
        self.th = th

    def __call__(self, batch):
        x, y, fn = batch
        x = repeat(x, "(n dim) -> n dim", n=self.n)
        y = reduce(y, "(n dim) -> n", n=self.n, reduction="mean") > self.th
        fn = [fn] * self.n
        return x, y, fn


class HotEncoder:
    """
    Class for converting labels to one-hot encoded format.

    Parameters
    ----------
    n_cls : int
        Number of classes for one-hot encoding.

    Methods
    -------
    __call__(batch: Tuple[torch.Tensor, torch.Tensor, Any]) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        Apply one-hot encoding to the labels in the batch.
    """

    def __init__(self, n_cls: int):
        self.n_cls = n_cls

    def __call__(self, batch):
        x, y, fn = batch
        return x, np.squeeze(np.eye(self.n_cls)[y.astype(int).reshape(-1)].transpose()), fn

    # def sanity_check(batch):
    #     # replaced with io_check method of CollateBase
    #     x, y, fn = batch
    #     x = torch.tensor(x, dtype=torch.float32)
    #     y = torch.tensor(y, dtype=torch.int32)
    #
    #     # Guarantee the x and y have 3-dimension shape.
    #     if len(x.shape) == 1:
    #         x = x.unsqueeze(0).unsqueeze(0)  # [dim] -> [1, 1, dim]
    #     elif len(x.shape) == 2:
    #         x = x.unsqueeze(1)  # [N, dim] -> [N, 1, dim]
    #     if len(y.shape) == 1:
    #         y = y.unsqueeze(1).unsqueeze(1)  # [dim] -> [1, 1, dim]
    #     elif len(y.shape) == 2:
    #         y = y.unsqueeze(0)  # [ch, dim] -> [N, 1, dim]
    #     return x, y, fn


def signal_filter(batch):
    # TODO: Fn wrapper with cutoff frequency fs
    nyq = fs / 2
    length = batch["x"].shape[-1]
    x = np.concatenate([batch["x"].squeeze()] * 3)
    x = signal.filtfilt(*signal.butter(2, [0.5 / nyq, 50 / nyq], btype="bandpass"), x, method="gust")
    x = signal.filtfilt(*signal.butter(2, [59.9 / nyq, 60.1 / nyq], btype="bandstop"), x, method="gust")
    batch["x"] = torch.tensor(x[length : 2 * length].reshape(1, -1).copy(), dtype=torch.float32)
    return batch


def split(batch):
    # TODO: Fn wrapper with overlap length overlapped
    raw_data = batch["x"].squeeze()
    batch["raw"] = raw_data.tolist()
    remain_length = (len(raw_data) - 2560) % (2560 - overlapped)
    if remain_length != 0:
        raw_data = F.pad(raw_data.unsqueeze(0), (0, 2560 - remain_length), mode="replicate").squeeze()
    splited = raw_data.unfold(dimension=0, size=2560, step=2560 - overlapped)

    batch["x"] = splited
    batch["fn"] = [batch["fn"]] * len(splited)
    return batch
