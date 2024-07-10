from collections import defaultdict
from typing import Union, Callable, List, Tuple, OrderedDict, overload, Iterator
from functools import wraps

import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader

from zae_engine.nn_night.blocks import UNetBlock
from zae_engine.models.builds import autoencoder
from zae_engine.models.converter import dim_converter
from zae_engine.trainer import Trainer
from zae_engine.schedulers import CosineAnnealingScheduler


class BeatCollateSeq:
    def __init__(self, sequence: Union[list, tuple] = tuple([None]), **kwargs):
        self.sequence = sequence
        self.xtype, self.ytype, self.info_type = ["x", "w", "onoff"], ["y", "r_loc"], ["fn", "rp"]
        self.resamp = kwargs["resamp"] if "resamp" in kwargs.keys() else 64
        self.zit = kwargs["zit"] if "zit" in kwargs.keys() else True
        self.overlapped = kwargs["overlapped"] if "overlapped" in kwargs.keys() else 560
        self.fs = 250

    def filter(self, batch):
        nyq = self.fs / 2
        length = batch["x"].shape[-1]
        x = np.concatenate([batch["x"].squeeze()] * 3)
        x = signal.filtfilt(*signal.butter(2, [0.5 / nyq, 50 / nyq], btype="bandpass"), x, method="gust")
        x = signal.filtfilt(*signal.butter(2, [59.9 / nyq, 60.1 / nyq], btype="bandstop"), x, method="gust")
        batch["x"] = torch.tensor(x[length : 2 * length].reshape(1, -1).copy(), dtype=torch.float32)
        return batch

    def split(self, batch):
        raw_data = batch["x"].squeeze()
        batch["raw"] = raw_data.tolist()
        remain_length = (len(raw_data) - 2560) % (2560 - self.overlapped)
        if remain_length != 0:
            raw_data = F.pad(raw_data.unsqueeze(0), (0, 2560 - remain_length), mode="replicate").squeeze()
        splited = raw_data.unfold(dimension=0, size=2560, step=2560 - self.overlapped)

        batch["x"] = splited
        batch["fn"] = [batch["fn"]] * len(splited)
        return batch

    def r_reg(self, batch):
        try:
            onoff = batch["onoff"]
        except KeyError:
            onoff = np.array(sanity_check(label_to_onoff(batch["y"].squeeze()), incomplete_only=True))

        if onoff is None:
            return {}

        try:
            assert len(onoff) == len(batch["rp"])
        except AssertionError:
            # For missing beat or R-gun at inference
            rp = torch.zeros(len(onoff))
            for i_onoff, (on, off, cls) in enumerate(onoff):
                i_on = np.searchsorted(batch["rp"], on)
                i_off = np.searchsorted(batch["rp"], off)
                if i_on + 1 == i_off:
                    rp[i_onoff] = batch["rp"][i_on]
            batch["rp"] = rp
        except (TypeError, AttributeError):
            pass
        except KeyError:
            batch["rp"] = None

        raw = batch["x"].squeeze()
        resampled, r_loc = [], []
        for i, (on, off, cls) in enumerate(onoff):
            if sum(np.isnan((on, off))):
                continue
            if self.zit:
                on, off = self.zitter(on, off)
            if off >= len(raw) - 1:
                off = -2
            on, off = int(on), int(off)
            chunk = raw[on : off + 1]
            if batch["rp"] is not None:
                r_loc.append((batch["rp"][i] - on) / (off - on))
            resampled.append(torch.tensor(signal.resample(chunk, self.resamp), dtype=torch.float32))

        batch["w"] = torch.stack(resampled, dim=0) if resampled else []
        batch["r_loc"] = torch.tensor(r_loc, dtype=torch.float32) if batch["rp"] is not None else None
        return batch

    def zitter(self, on, off):
        if off - on > 10:
            on += np.random.randint(-3, 4)
            off += np.random.randint(-3, 4)
        else:
            on += np.random.randint(-3, 2)
            off += np.random.randint(-1, 4)
        return max(0, on), off

    def accumulate(self, batches: Union[Tuple, List]):
        accumulate_dict = defaultdict(list)
        # Convert a list of dictionaries per data to a batch dictionary with list-type values.
        for b in batches:
            for k, v in b.items():
                if isinstance(v, list):
                    accumulate_dict[k] += v
                else:
                    accumulate_dict[k].append(v)
        for k, v in accumulate_dict.items():
            try:
                if set(v) == {None}:
                    accumulate_dict[k] = None
                elif k in self.info_type:
                    pass
                elif k in self.xtype:
                    accumulate_dict[k] = torch.cat(v, dim=0).unsqueeze(1) if v else []
                else:
                    accumulate_dict[k] = torch.cat(v, dim=0).squeeze()
            except TypeError:
                pass
        return accumulate_dict

    def __call__(self, batch: dict or list):
        batches = []
        for b in batch:
            b = self.filter(b)
            for seq in self.sequence:
                if seq == "r_reg":
                    b = self.r_reg(b)
                elif seq == "split":
                    b = self.split(b)
                else:
                    pass
            batches.append(b)
        batches = self.accumulate(batches)
        return batches

    def wrap(self, func: Callable = None):
        if func is None:
            func = self.__call__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func


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
        inference_dataset, batch_size=1, shuffle=False, collate_fn=BeatCollateSeq(sequence=["split"]).wrap()
    )

    # --------------------------------- Inference & Postprocess @ stage 1 --------------------------------- #
    model = autoencoder.AutoEncoder(
        block=UNetBlock, ch_in=1, ch_out=4, width=16, layers=[1, 1, 1, 1], skip_connect=True
    )
    cvtr = dim_converter.DimConverter(model)
    model = cvtr.convert("2d -> 1d")
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = CosineAnnealingScheduler(optimizer, total_iters=100)
    trainer1 = Trainer_stg1(model=model, device=device, mode="test", optimizer=optimizer, scheduler=scheduler)
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


if __name__ == "__main__":
    x = np.zeros(2048000)
    core(x)
