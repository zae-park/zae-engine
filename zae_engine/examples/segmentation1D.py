import logging
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from zae_engine.nn_night.blocks import UNetBlock
from zae_engine.models.builds import autoencoder
from zae_engine.models.converter import dim_converter
from zae_engine.trainer import Trainer
from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.data.collate import CollateBase
from zae_engine.data.collate.modules import Spliter

# ------------------------------- Custom ----------------------------------- #
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.zeros_like(self.x)
        self.fn = ["example"] * x.shape[0]
        print(f"\t # of data: %d" % len(self.x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx].unsqueeze(0), "y": self.y[idx], "fn": self.fn[idx]}


class InferenceTrainer(Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(InferenceTrainer, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 32

    def train_step(self, batch: dict):
        pass

    def test_step(self, batch: dict):
        x = batch["x"]
        mini_x = x.split(self.mini_batch_size, dim=0)
        out = torch.cat([self.model(m_x) for m_x in mini_x])
        return {"loss": 0, "output": out}


# ------------------------------- Core ----------------------------------- #


def core(x: Union[np.ndarray, torch.Tensor]) -> list:
    """
    Core function to perform inference on a given 1-D input array using a predefined model and data pipeline.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        The input 1-D array to be processed.

    Returns
    -------
    np.ndarray
        The result of the inference process as a numpy array with the predicted class indices.

    Raises
    ------
    AssertionError
        If the input array is not 1-D.

    Example
    -------
    >>> x = np.zeros(2048000)
    >>> result = core(x)
    >>> print(result)

    Notes
    -----
    This function performs the following steps:
    1. Checks if the input array is 1-D.
    2. Sets up the device for computation (CPU or GPU).
    3. Initializes the data pipeline, including dataset and data loader with necessary preprocessing steps.
    4. Sets up the model, optimizer, and learning rate scheduler.
    5. Runs the inference process using the model and returns the predicted class indices.

    The model used is an autoencoder with a UNetBlock structure, converted from 2D to 1D. The data is split into
    smaller segments using the Spliter class before being fed into the model. The inference is performed in batches
    to handle large input arrays efficiently.
    """

    assert len(x.shape) == 1, f"Expect 1-D array, but receive {len(x.shape)}-D array."
    if not x.size:
        return []
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # --------------------------------- Data Pipeline --------------------------------- #
        inference_dataset = InferenceDataset(x=x.reshape(-1, 2048))

        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["fn"])
        collator.set_batch(inference_dataset[0])

        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collator.wrap())

        # --------------------------------- Setting --------------------------------- #
        model = autoencoder.AutoEncoder(block=UNetBlock, ch_in=1, ch_out=4, width=16, layers=[1] * 4, skip_connect=True)
        model = dim_converter.DimConverter(model).convert("2d -> 1d")
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = CosineAnnealingScheduler(optimizer, total_iters=100)
        trainer1 = InferenceTrainer(model=model, device=device, mode="test", optimizer=optimizer, scheduler=scheduler)

        # --------------------------------- Inference --------------------------------- #
        return np.concatenate(trainer1.inference(inference_loader)).argmax(1).tolist()

    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        raise


if __name__ == "__main__":
    x = np.zeros(20480)
    core(x)
