import logging
from typing import Union, Dict, Optional, List, Tuple

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


def core(x: np.ndarray, batch_size: int):
    assert len(x.shape) < 3, f"Expect less than 3-D array, but receive {len(x.shape)}-D array."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # --------------------------------- Data Pipeline --------------------------------- #
        inference_dataset = InferenceDataset(x)

        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=[])
        collator.set_batch(inference_dataset[0])
        collator.add_fn("filtering", SignalFilter(fs=250, method="bandpass", lowcut=0.5, highcut=50))
        collator.add_fn("scaling", SignalScaler())
        collator.add_fn("chunk", Chunk(n=2500))
        collator.add_fn("hot", HotEncoder(n_cls=7))

        inference_loader = DataLoader(dataset=inference_dataset, batch_size=batch_size, collate_fn=collator.wrap())

        # --------------------------------- Setting --------------------------------- #
        model = CNNBase(BasicBlock, 1, 9, 16, [2, 2, 2, 2])
        model = dim_converter.DimConverter(model).convert("2d -> 1d")
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = CosineAnnealingScheduler(optimizer, total_iters=100)
        trainer = InferenceTrainer(model, device, "test", optimizer, scheduler)

        # --------------------------------- Inference --------------------------------- #
        result = trainer.inference(inference_loader)
        result = np.concatenate(result).reshape(len(inference_dataset), -1)
        return result
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    x = np.zeros(2048000)
    core(x)
