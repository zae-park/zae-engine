from typing import Dict, Any
import numpy as np
import torch
from torch.nn import functional as F
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class UnifiedChunker:
    """
    Unified Chunker Class: Splits input data into chunks based on the tensor's dimensionality.

    Parameters
    ----------
    chunk_size : int
        The size of each chunk.
    overlap : int, optional
        The overlap size between consecutive chunks. Default is 0.
    """

    def __init__(self, chunk_size: int, overlap: int = 0):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Splits the batch into chunks.

        Parameters
        ----------
        batch : Dict[str, Any]
            - 'x': torch.Tensor of shape (sequence_length,) or (batch_size, sequence_length)
            - 'y': Optional[torch.Tensor]
            - 'fn': Optional[Any]

        Returns
        -------
        Dict[str, Any]
            The batch after splitting into chunks.
        """
        if "x" not in batch:
            raise KeyError("Batch must contain 'x' key.")

        x = batch["x"]
        y = batch.get("y", None)
        fn = batch.get("fn", None)

        if not isinstance(x, torch.Tensor):
            raise TypeError("'x' must be a torch.Tensor.")

        if x.dim() == 1:
            # Handle 1D tensor
            chunks = self._split_1d(x)
        elif x.dim() == 2:
            # Handle 2D tensor (split each sample in the batch)
            chunks = self._split_2d(x)
        else:
            raise ValueError("Unsupported tensor dimension. Only 1D and 2D tensors are supported.")

        # Process 'y' if it exists (e.g., repeat or average based on chunking)
        if y is not None:
            y_chunks = self._process_y(y, x.dim())
            batch["y"] = y_chunks

        # Process 'fn' if it exists (e.g., repeat to match chunks)
        if fn is not None:
            fn_chunks = self._process_fn(fn, x.dim())
            batch["fn"] = fn_chunks

        # Update 'x' in the batch with the chunks
        batch["x"] = chunks

        return batch

    def _split_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits a 1D tensor into chunks.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (sequence_length,)

        Returns
        -------
        torch.Tensor
            The tensor split into chunks. Shape: (num_chunks, chunk_size)
        """
        x_np = x.cpu().numpy()
        chunks = self._create_chunks(x_np, self.chunk_size, self.overlap)
        chunks_tensor = torch.tensor(chunks, dtype=x.dtype, device=x.device)
        return chunks_tensor

    def _split_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits a 2D tensor into chunks for each sample in the batch.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, sequence_length)

        Returns
        -------
        torch.Tensor
            The tensor split into chunks. Shape: (batch_size * num_chunks, chunk_size)
        """
        batch_size, seq_length = x.shape
        x_np = x.cpu().numpy()
        chunks = []
        for i in range(batch_size):
            sample_chunks = self._create_chunks(x_np[i], self.chunk_size, self.overlap)
            chunks.append(sample_chunks)
        chunks_np = np.vstack(chunks)
        chunks_tensor = torch.tensor(chunks_np, dtype=x.dtype, device=x.device)
        return chunks_tensor

    def _create_chunks(self, data: np.ndarray, chunk_size: int, overlap: int) -> np.ndarray:
        """
        Splits data into chunks.

        Parameters
        ----------
        data : np.ndarray
            1D data array.
        chunk_size : int
            Size of each chunk.
        overlap : int
            Overlap size between chunks.

        Returns
        -------
        np.ndarray
            Array of chunks. Shape: (num_chunks, chunk_size)
        """
        step = chunk_size - overlap
        num_chunks = (len(data) - overlap) // step
        if (len(data) - overlap) % step != 0:
            num_chunks += 1  # Add an extra chunk for remaining data

        chunks = []
        for i in range(num_chunks):
            start = i * step
            end = start + chunk_size
            chunk = data[start:end]
            if len(chunk) < chunk_size:
                # Pad with the last value if chunk is incomplete
                pad_width = chunk_size - len(chunk)
                chunk = np.pad(chunk, (0, pad_width), mode="edge")
            chunks.append(chunk)
        return np.array(chunks)

    def _process_y(self, y: torch.Tensor, x_dim: int) -> torch.Tensor:
        """
        Processes 'y' values to match the chunked 'x'.

        Parameters
        ----------
        y : torch.Tensor
            Original 'y' values. Shape: (batch_size,) or (sequence_length,)
        x_dim : int
            Dimensionality of 'x'.

        Returns
        -------
        torch.Tensor
            Processed 'y' values.
        """
        if x_dim == 1:
            # For 1D tensor, split 'y' similarly to 'x'
            y_np = y.cpu().numpy()
            chunks = self._create_chunks(y_np, self.chunk_size, self.overlap)
            y_chunks = torch.tensor(chunks, dtype=y.dtype, device=y.device)
            return y_chunks
        elif x_dim == 2:
            # For 2D tensor, repeat 'y' to match the number of chunks
            y_np = y.cpu().numpy()
            y_repeated = np.repeat(y_np, self._get_num_chunks(y_np.shape[0]))
            y_chunks = torch.tensor(y_repeated, dtype=y.dtype, device=y.device)
            return y_chunks
        else:
            raise ValueError("Unsupported tensor dimension for 'y' processing.")

    def _process_fn(self, fn: Any, x_dim: int) -> Any:
        """
        Processes 'fn' values to match the chunked 'x'.

        Parameters
        ----------
        fn : Any
            Original 'fn' value.
        x_dim : int
            Dimensionality of 'x'.

        Returns
        -------
        Any
            Processed 'fn' values.
        """
        if x_dim == 1:
            # For 1D tensor, repeat 'fn' as a list
            num_chunks = self._get_num_chunks(len(fn)) if isinstance(fn, list) else 1
            fn_chunks = [fn] * num_chunks
            return fn_chunks
        elif x_dim == 2:
            # For 2D tensor, repeat each 'fn' for its respective chunks
            if isinstance(fn, list):
                fn_chunks = [item for item in fn for _ in range(self._get_num_chunks(len(fn)))]
            else:
                fn_chunks = [fn] * self._get_num_chunks(len(fn))
            return fn_chunks
        else:
            raise ValueError("Unsupported tensor dimension for 'fn' processing.")

    def _get_num_chunks(self, length: int) -> int:
        """
        Calculates the number of chunks for a given data length.

        Parameters
        ----------
        length : int
            Length of the original data.

        Returns
        -------
        int
            Number of chunks.
        """
        step = self.chunk_size - self.overlap
        num_chunks = (length - self.overlap) // step
        if (length - self.overlap) % step != 0:
            num_chunks += 1
        return num_chunks


class Chunk:
    """
    Class for reshaping the 'x' tensor within a batch.

    Parameters
    ----------
    n : int
        The size to reshape the 'x' tensor.

    Expected Input Shapes
    ---------------------
    batch: Dict[str, Any]
        - 'x': torch.Tensor of shape (batch_size, n * some_integer)

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Reshapes the 'x' tensor in the batch.
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

    Expected Input Shapes
    ---------------------
    batch: Dict[str, Any]
        - 'y': torch.Tensor of shape (batch_size,) or (batch_size, ...)

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Applies one-hot encoding to the labels in the batch.
    """

    def __init__(self, n_cls: int):
        if n_cls <= 0:
            raise ValueError("Number of classes 'n_cls' must be positive.")
        self.n_cls = n_cls

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "y" not in batch:
            raise KeyError("Batch must contain 'y' key.")

        y = batch["y"]

        if y.dim() == 1:
            batch["y_hot"] = torch.eye(self.n_cls, device=y.device)[y.long()]
        else:
            # Apply one-hot encoding to the last dimension if y is multi-dimensional
            batch["y_hot"] = torch.eye(self.n_cls, device=y.device)[y.long()]

        return batch


class SignalFilter:
    """
    Class for filtering signals within a batch.

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

    Expected Input Shapes
    ---------------------
    batch: Dict[str, Any]
        - 'x': torch.Tensor of shape (sequence_length,) or (batch_size, sequence_length)

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Applies the specified filter to the signals in the batch.
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
        batch["x"] = torch.tensor(x_filtered.copy(), dtype=torch.float32).unsqueeze(0).to(x.device)

        return batch


class Spliter:
    """
    Class for splitting signals within a batch with overlapping.

    Parameters
    ----------
    chunk_size : int, default=2560
        The size of each chunk after splitting.
    overlapped : int, default=0
        The number of overlapping samples between adjacent segments.

    Expected Input Shapes
    ---------------------
    batch: Dict[str, Any]
        - 'x': torch.Tensor of shape (sequence_length,)

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Splits the signal in the batch with the specified overlap.
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
    Class for scaling signals within a batch using MinMaxScaler.

    Parameters
    ----------
    None

    Expected Input Shapes
    ---------------------
    batch: Dict[str, Any]
        - 'x': torch.Tensor of shape (features,) or (batch_size, features)

    Methods
    -------
    __call__(batch: Dict[str, Any]) -> Dict[str, Any]:
        Applies MinMax scaling to the signals in the batch.
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

        # If original input was 1D, flatten the scaled array
        if x_np.shape[1] == 1:
            x_scaled = x_scaled.flatten()

        # Convert back to torch.Tensor and preserve device
        batch["x"] = torch.tensor(x_scaled, dtype=torch.float32, device=x.device)

        return batch
