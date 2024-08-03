from typing import Dict, Union

import numpy as np
import torch
from torch.nn import functional as F
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from einops import repeat, reduce


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
        x, y, fn = batch["x"], batch["y"], batch["fn"]
        x = repeat(x, "(n dim) -> n dim", n=self.n)
        y = reduce(y, "(n dim) -> n", n=self.n, reduction="mean") > self.th
        fn = [fn] * self.n
        return x, y, fn


class Chunk:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, batch: dict) -> Dict:
        x = batch["x"]
        modulo = x.shape[-1] % self.n
        if modulo != 0:
            x = x[:, :-modulo]
        batch["x"] = x.reshape(-1, self.n)
        return batch


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
        y = batch["y"]
        batch["y"] = torch.eye(self.n_cls)[y.int()]
        return batch


class SignalFilter:
    """
    Class for filtering signals in the batch.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal.
    method : str
        Filtering method to apply. Options are 'bandpass', 'bandstop', 'lowpass', 'highpass'.
    lowcut : float, optional
        Low cut-off frequency for bandpass and bandstop filters.
    highcut : float, optional
        High cut-off frequency for bandpass and bandstop filters.
    cutoff : float, optional
        Cut-off frequency for lowpass and highpass filters.

    Methods
    -------
    __call__(batch: dict) -> dict:
        Apply the specified filter to the signal in the batch.
    """

    def __init__(self, fs: float, method: str, lowcut: float = None, highcut: float = None, cutoff: float = None):
        self.fs = fs
        self.method = method
        self.lowcut = lowcut
        self.highcut = highcut
        self.cutoff = cutoff

    def __call__(self, batch: Dict[str, Union[torch.Tensor, list]]) -> Dict:
        nyq = self.fs / 2
        x = batch["x"].squeeze().numpy()

        if self.method == "bandpass":
            if self.lowcut is None or self.highcut is None:
                raise ValueError("Lowcut and highcut frequencies must be specified for bandpass filter.")
            b, a = signal.butter(2, [self.lowcut / nyq, self.highcut / nyq], btype="bandpass")
        elif self.method == "bandstop":
            if self.lowcut is None or self.highcut is None:
                raise ValueError("Lowcut and highcut frequencies must be specified for bandstop filter.")
            b, a = signal.butter(2, [self.lowcut / nyq, self.highcut / nyq], btype="bandstop")
        elif self.method == "lowpass":
            if self.cutoff is None:
                raise ValueError("Cutoff frequency must be specified for lowpass filter.")
            b, a = signal.butter(2, self.cutoff / nyq, btype="low")
        elif self.method == "highpass":
            if self.cutoff is None:
                raise ValueError("Cutoff frequency must be specified for highpass filter.")
            b, a = signal.butter(2, self.cutoff / nyq, btype="high")
        else:
            raise ValueError(
                f"Invalid method: {self.method}. Choose from 'bandpass', 'bandstop', 'lowpass', 'highpass'."
            )

        x = signal.filtfilt(b, a, np.concatenate([x] * 3), method="gust")
        batch["x"] = torch.tensor(x[len(x) // 3 : 2 * len(x) // 3].copy(), dtype=torch.float32).unsqueeze(0)
        return batch


class Spliter:
    """
    Class for splitting signals in the batch with overlapping.

    Parameters
    ----------
    overlapped : int
        The number of overlapping samples between adjacent segments.

    Methods
    -------
    __call__(batch: dict) -> dict:
        Split the signal in the batch with the specified overlap.
    """

    def __init__(self, overlapped: int):
        self.overlapped = overlapped

    def __call__(self, batch: dict) -> dict:
        raw_data = batch["x"].squeeze()
        batch["raw"] = raw_data.tolist()
        remain_length = (len(raw_data) - 2560) % (2560 - self.overlapped)
        if remain_length != 0:
            raw_data = F.pad(raw_data.unsqueeze(0), (0, 2560 - remain_length), mode="replicate").squeeze()
        splited = raw_data.unfold(dimension=0, size=2560, step=2560 - self.overlapped)

        batch["x"] = splited
        return batch


class SignalScaler:
    """
    Class for scaling signals in the batch.

    Parameters
    ----------
    None

    Methods
    -------
    __call__(batch: dict) -> dict:
        Apply MinMax scaling to the signal in the batch.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def __call__(self, batch):
        x = batch["x"].numpy() if isinstance(batch["x"], torch.Tensor) else batch["x"]
        scaled_batch = []

        for subset_x in x:
            self.scaler.fit(np.expand_dims(subset_x, 1))
            subset_x = self.scaler.transform(np.expand_dims(subset_x, 1)).squeeze()
            scaled_batch.append(subset_x)

        batch["x"] = torch.tensor(np.array(scaled_batch), dtype=torch.float32)
        return batch
