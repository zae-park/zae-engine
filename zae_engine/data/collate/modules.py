from typing import Union, Callable, List, Tuple, OrderedDict, overload, Iterator
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps

import numpy as np
from scipy import signal
import torch
from torch.nn import functional as F
from einops import repeat, reduce

from zae_engine.operation import label_to_onoff


def chunk(batch):
    # TODO: Fn wrapper with chunk size n & th
    x, y, fn = batch
    x = repeat(x, "(n dim) -> n dim", n=n)
    y = reduce(y, "(n dim) -> n", n=n, reduction="mean") > th
    fn = [fn] * n
    return x, y, fn


def hot(batch):
    # TODO: Fn wrapper with num_classes n_cls
    x, y, fn = batch
    return x, np.squeeze(np.eye(n_cls)[y.astype(int).reshape(-1)].transpose()), fn

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
