from typing import Dict, Union, List, Any
import numpy as np
import torch
from torch.nn import functional as F
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from einops import repeat, reduce
import logging

logger = logging.getLogger(__name__)


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
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Chunk the data in the batch.
    """

    def __init__(self, n: int, th: float):
        self.n = n
        self.th = th

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch must contain 'x' and 'y' keys.")

        x = batch["x"]
        y = batch["y"]
        fn = batch.get("fn", None)  # Handle missing 'fn' key gracefully

        # Repeat 'x' tensor n times
        x = repeat(x, "(n dim) -> n dim", n=self.n)

        # Reduce 'y' tensor by mean and apply threshold
        y = (reduce(y, "(n dim) -> n", n=self.n, reduction="mean") > self.th).float()

        # Repeat 'fn' list or set to None if 'fn' is not present
        fn = [fn] * self.n if fn is not None else [None] * self.n

        # Update the batch dictionary
        batch["x"] = x
        batch["y"] = y
        batch["fn"] = fn

        return batch


class Chunk:
    """
    Class for reshaping the 'x' tensor in the batch.

    Parameters
    ----------
    n : int
        The size to reshape the 'x' tensor.

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Reshape the 'x' tensor in the batch.
    """

    def __init__(self, n: int):
        self.n = n

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "x" not in batch:
            raise KeyError("Batch must contain 'x' key.")

        x = batch["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError("'x' must be a torch.Tensor.")

        if x.dim() < 2:
            raise ValueError("'x' tensor must have at least 2 dimensions.")

        modulo = x.shape[-1] % self.n
        if modulo != 0:
            x = x[:, :-modulo]
        x = x.reshape(-1, self.n)

        batch["x"] = x
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
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Apply one-hot encoding to the labels in the batch.
    """

    def __init__(self, n_cls: int):
        self.n_cls = n_cls

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "y" not in batch:
            raise KeyError("Batch must contain 'y' key.")

        y = batch["y"]

        if y.dim() == 1:
            batch["y"] = torch.eye(self.n_cls, device=y.device)[y.long()]
        else:
            # Apply one-hot encoding to the last dimension if y is multi-dimensional
            batch["y"] = torch.eye(self.n_cls, device=y.device)[y.long()]

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
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Apply the specified filter to the signal in the batch.
    """

    def __init__(self, fs: float, method: str, lowcut: float = None, highcut: float = None, cutoff: float = None):
        self.fs = fs
        self.method = method
        self.lowcut = lowcut
        self.highcut = highcut
        self.cutoff = cutoff

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "x" not in batch:
            raise KeyError("Batch must contain 'x' key.")

        x = batch["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError("'x' must be a torch.Tensor.")

        if x.dim() < 1:
            raise ValueError("'x' tensor must have at least 1 dimension.")

        nyq = self.fs / 2
        x_np = x.squeeze().cpu().numpy()

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

        # Apply filter with padding to minimize edge effects
        try:
            x_filtered = signal.filtfilt(b, a, np.concatenate([x_np] * 3), method="gust")
            # Remove padding
            x_filtered = x_filtered[len(x_np) : 2 * len(x_np)]
        except Exception as e:
            logger.error(f"Error during filtering: {e}")
            raise TypeError(f"Error during filtering: {e}") from e

        # Convert back to torch.Tensor and preserve device
        batch["x"] = torch.tensor(x_filtered, dtype=torch.float32).unsqueeze(0).to(x.device)

        return batch


class Spliter:
    """
    Class for splitting signals in the batch with overlapping.

    Parameters
    ----------
    chunk_size : int, default=2560
        The size of each chunk after splitting.
    overlapped : int, default=0
        The number of overlapping samples between adjacent segments.

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Split the signal in the batch with the specified overlap.
    """

    def __init__(self, chunk_size: int = 2560, overlapped: int = 0):
        self.chunk_size = chunk_size
        self.overlapped = overlapped

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "x" not in batch:
            raise KeyError("Batch must contain 'x' key.")

        x = batch["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError("'x' must be a torch.Tensor.")

        if x.dim() < 1:
            raise ValueError("'x' tensor must have at least 1 dimension.")

        raw_data = x.squeeze()

        # Calculate step size
        step = self.chunk_size - self.overlapped
        if step <= 0:
            raise ValueError("overlapped must be less than chunk_size.")

        total_length = len(raw_data)
        remain_length = (total_length - self.chunk_size) % step
        if remain_length != 0:
            padding_length = step - remain_length
            raw_data = F.pad(raw_data.unsqueeze(0), (0, padding_length), mode="replicate").squeeze()

        # Unfold to create chunks
        splited = raw_data.unfold(dimension=0, size=self.chunk_size, step=step)

        # Update 'x' with splitted chunks
        batch["x"] = splited

        return batch


class SignalScaler:
    """
    Class for scaling signals in the batch using MinMaxScaler.

    Parameters
    ----------
    None

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Apply MinMax scaling to the signal in the batch.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "x" not in batch:
            raise KeyError("Batch must contain 'x' key.")

        x = batch["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError("'x' must be a torch.Tensor.")

        # Convert to numpy for scaling
        x_np = x.cpu().numpy()

        # Ensure x has at least 2 dimensions for scaler
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        elif x_np.ndim > 2:
            raise ValueError("'x' tensor must have 1 or 2 dimensions.")

        # Fit and transform
        self.scaler.fit(x_np)
        x_scaled = self.scaler.transform(x_np)

        # Convert back to torch.Tensor and preserve device
        batch["x"] = torch.tensor(x_scaled, dtype=torch.float32, device=x.device)

        return batch
