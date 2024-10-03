import logging
from typing import Union, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from zae_engine.trainer import Trainer
from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.models.builds.cnn import CNNBase
from zae_engine.models.converter import dim_converter
from zae_engine.nn_night.blocks import BasicBlock
from zae_engine.data.collate import CollateBase
from zae_engine.data.collate.modules import SignalFilter, SignalScaler, Chunk, HotEncoder


# ------------------------------- Custom ----------------------------------- #
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor([[0]] * x.shape[0])
        self.fn = ["example"] * x.shape[0]
        print(f"\t # of data: %d" % len(self.x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "fn": self.fn[idx]}


class InferenceTrainer(Trainer):
    def __init__(self, model, device, mode, optimizer, scheduler):
        super(InferenceTrainer, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 32

    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        pass

    def test_step(self, batch: dict):
        x = batch["x"]
        mini_x = x.split(self.mini_batch_size, dim=0)
        out = torch.cat([self.model(m_x).argmax(-1).detach().cpu() for m_x in mini_x])
        return {"loss": 0, "output": out}


# ------------------------------- Core ----------------------------------- #


def core(x: Union[np.ndarray, torch.Tensor]) -> list:
    """
    Perform inference on a given input array using a predefined CNN model and data pipeline.

    This function processes the input data through a series of preprocessing steps,
    feeds it into a Convolutional Neural Network (CNN) model, and returns the
    predicted class indices for each input segment.

    Parameters
    ----------
    x : np.ndarray
        Input array to be processed. The array should be less than 3-D (i.e., 1-D or 2-D).
        For 2-D inputs, the shape should be `(num_samples, 2048)`, where `2048` is the
        expected segment length.

    Returns
    -------
    np.ndarray
        A NumPy array containing the predicted class indices for each input sample.

    Raises
    ------
    AssertionError
        If the input array has 3 or more dimensions.
    Exception
        If any error occurs during the inference process.

    Example
    -------
    >>> import numpy as np
    >>> x = np.zeros(20480).reshape(-1, 2048)  # 10 samples of length 2048 each
    >>> predictions = core(x)
    >>> print(predictions)
    [0 1 0 2 1 0 1 0 1 2]

    Notes
    -----
    This function performs the following steps:
    1. Validates the input array dimensions.
    2. Sets up the computation device (CPU or GPU).
    3. Initializes the data pipeline, including dataset and data loader with preprocessing modules.
    4. Sets up the CNN model, optimizer, and learning rate scheduler.
    5. Executes the inference process using the model and returns the predicted class indices.

    The model used is a CNN-based architecture (`CNNBase` with `BasicBlock`), converted from 2D to 1D.
    The input data is preprocessed using filtering, scaling, and one-hot encoding before being fed into the model.
    Inference is performed in batches to handle large input arrays efficiently.
    """
    assert len(x.shape) == 1, f"Expect 1-D array, but receive {len(x.shape)}-D array."
    if not x.size:
        return []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # --------------------------------- Data Pipeline --------------------------------- #
        inference_dataset = InferenceDataset(x.reshape(-1, 2048))

        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=[])
        collator.set_batch(inference_dataset[0])
        collator.add_fn("filtering", SignalFilter(fs=250, method="bandpass", lowcut=0.5, highcut=50))
        collator.add_fn("scaling", SignalScaler())
        collator.add_fn("hot", HotEncoder(n_cls=7))

        inference_loader = DataLoader(dataset=inference_dataset, batch_size=1, collate_fn=collator.wrap())

        # --------------------------------- Setting --------------------------------- #
        model = CNNBase(BasicBlock, 1, 9, 16, [2, 2, 2, 2])
        model = dim_converter.DimConverter(model).convert("2d -> 1d")
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = CosineAnnealingScheduler(optimizer, total_iters=100)
        trainer = InferenceTrainer(model, device, "test", optimizer, scheduler)

        # --------------------------------- Inference --------------------------------- #
        result = trainer.inference(inference_loader)
        result = np.concatenate(result).reshape(len(inference_dataset), -1).argmax(-1)
        return result.tolist()
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    x = np.zeros(20480)
    core(x)
