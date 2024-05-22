from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..models import beat_segmentation
from ..trainer import Trainer
from ..data.collate.collate import BeatCollateSeq as Col


def core(x: Union[np.ndarray, torch.Tensor]):
    """
    Detect beat and its class for a given waveform.
    :param x: List of waveforms.
    :return: Onoff matrix consists of [[on, off, cls, rpeak] x beats] for each x.
    """
    assert len(x.shape) == 1, f"Expect 1-D array, but receive {len(x.shape)}-D array."
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    inference_dataset = ECG_dataset(x=x.reshape(1, -1))
    inference_loader = DataLoader(
        inference_dataset, batch_size=1, shuffle=False, collate_fn=Col(sequence=["split"]).wrap()
    )

    # --------------------------------- Inference & Postprocess @ stage 1 --------------------------------- #
    model = beat_segmentation(True)
    trainer1 = Trainer_stg1(model=model, device=device, mode="test")
    return np.concatenate(trainer1.inference(inference_loader)).argmax(1)


# --------------------------------- Dataset --------------------------------- #
class ECG_dataset(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.x = torch.tensor(kwargs["x"], dtype=torch.float32)
        self.y = torch.tensor(kwargs["y"], dtype=torch.long) if self.attr("y") is not None else None
        self.fn = kwargs["fn"] if self.attr("fn") is not None else None
        self.rp = kwargs["rp"] if self.attr("rp") is not None else None
        self.onoff = kwargs["onoff"] if self.attr("onoff") is not None else None

        self.mode = kwargs["mode"] if "mode" in kwargs.keys() else ""
        print(f"\t # of {self.mode} data_pipeline: %d" % len(self.x))

    def __len__(self):
        return len(self.x)

    def attr(self, var_name):
        if var_name in self.kwargs.keys():
            return self.kwargs[var_name]

    def __getitem__(self, idx):
        batch_dict = defaultdict(None)
        batch_dict["x"] = self.x[idx].unsqueeze(0)
        batch_dict["y"] = self.y[idx].unsqueeze(0) if self.y is not None else None
        batch_dict["fn"] = self.fn[idx] if self.fn is not None else None
        if self.rp is not None:
            batch_dict["rp"] = torch.tensor(self.rp[idx], dtype=torch.long)
        if self.onoff:
            batch_dict["onoff"] = self.onoff[idx] if self.onoff is not None else None
        return batch_dict


class Trainer_stg1(Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(Trainer_stg1, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 32

    def train_step(self, batch: dict):
        pass

    def test_step(self, batch: dict):
        x = batch["x"]
        mini_x = x.split(self.mini_batch_size, dim=0)
        out = torch.cat([self.model(m_x) for m_x in mini_x])
        return {"loss": 0, "output": out}
