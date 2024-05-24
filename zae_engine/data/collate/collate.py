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


class CollatorBase(ABC):
    _fn: OrderedDict[str, Callable]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Callable) -> None: ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Callable]") -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_fn(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_fn(str(idx), module)
        self.sample_batch = {}

    def __len__(self) -> int:
        return len(self._fn)

    def __iter__(self) -> Iterator:
        return iter(self._fn.values())

    def io_check(self, fn: Callable) -> None:
        keys = self.sample_batch.keys()
        updated = fn(self.sample_batch)
        assert isinstance(updated, type(self.sample_batch))
        assert set(keys).issubset(updated.keys())

    def set_batch(self, batch: Union[dict, OrderedDict]) -> None:
        self.sample_batch = batch

    def add_fn(self, name: str, fn: Callable) -> None:
        self.io_check(fn)
        self._fn[name] = fn

    def __call__(self, batch: Union[dict, OrderedDict]) -> Union[dict, OrderedDict]:
        for fn in self:
            batch = fn(batch)

        return batch


class Collate_seq(ABC):
    """
    TBD.

    # usage
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.run.batch_train, shuffle=True,
                                                   pin_memory=True, collate_fn=Collate_seq(sequence=seq_list))
    """

    def __init__(self, sequence: Union[list, tuple] = tuple([None]), n: int = 10, n_cls: int = 2, th: float = 0.5):
        self.sequence = sequence
        self.n = n
        self.n_cls = n_cls
        self.th = th

    @abstractmethod
    def chunk(self, batch):
        x, y, fn = batch
        x = repeat(x, "(n dim) -> n dim", n=self.n)
        y = reduce(y, "(n dim) -> n", n=self.n, reduction="mean") > self.th
        fn = [fn] * self.n
        return x, y, fn

    @abstractmethod
    def hot(self, batch):
        x, y, fn = batch
        return x, np.squeeze(np.eye(self.n_cls)[y.astype(int).reshape(-1)].transpose()), fn

        # hot = F.one_hot(batch[1], num_classes=self.n_cls)
        # return tuple([batch[0], hot, batch[2]])

    @staticmethod
    def sanity_check(batch):
        x, y, fn = batch
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int32)

        # Guarantee the x and y have 3-dimension shape.
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [dim] -> [1, 1, dim]
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)  # [N, dim] -> [N, 1, dim]
        if len(y.shape) == 1:
            y = y.unsqueeze(1).unsqueeze(1)  # [dim] -> [1, 1, dim]
        elif len(y.shape) == 2:
            y = y.unsqueeze(0)  # [ch, dim] -> [N, 1, dim]
        return x, y, fn

    def __call__(self, batch):
        x, y, fn = [], [], []
        for b in batch:
            for seq in self.sequence:
                if seq == "chunk":
                    b = self.chunk(b)
                elif seq == "hot":
                    b = self.hot(b)
                else:
                    pass
            b = self.sanity_check(b)
            x.append(b[0]), y.append(b[1]), fn.append(b[2])
        return torch.cat(x), torch.cat(y), fn


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


class Collator(Collate_seq):
    def __init__(self, feat_dim: int = 27, max_length: int = 512, *logics: str):
        super().__init__(sequence=logics)
        self.logics = logics
        self.max_length = max_length
        self.func_map = {
            "sort": self.sort,
            "event": self.events,
            "time": self.time_stamp,
            "purchase": self.purchase,
            "hot": self.hot,
            "emb": self.emb,
            "loc": self.relative_size,
            "dict": self.as_dict,
        }
        self.url_eye = torch.eye(14, dtype=torch.float32)
        self.event_map = {"null": 0, "pageview": 1, "click": 2, "purchase": 3}
        self.event_eye = torch.eye(len(self.event_map), dtype=torch.float32)

        self.canvas = torch.zeros(self.max_length, feat_dim)

    def as_dict(self, mini_batch):
        return mini_batch._asdict()

    def chunk(self, mini_batch):
        return mini_batch

    def sort(self, mini_batch):
        sorted_cols = [k for k in mini_batch.keys() if (k.endswith("list") and not k.startswith("request"))]
        dt2int = np.array(mini_batch["tracking_date_list"], dtype=int)
        sort_idx = np.argsort(dt2int)

        mini_batch["tracking_date_list"] = dt2int[sort_idx]
        for i, c in enumerate(sorted_cols):
            try:
                mini_batch[c] = mini_batch[c][sort_idx]
            except:
                mini_batch[c] = np.zeros_like(mini_batch["event_list"])[sort_idx]
        return mini_batch

    def hot(self, mini_batch):
        # subsequence method for url_label

        mini_batch["url_sanity"] = torch.ones(1)
        label_cnt = len(mini_batch["label_list"])
        event_cnt = len(mini_batch["event_list"])
        if label_cnt > event_cnt:
            mini_batch["label_list"] = [0] * len(mini_batch["event_list"])
            mini_batch["url_sanity"] = torch.zeros(1)
        else:
            pageview_cnt = mini_batch["event_list"].count("pageview")
            if pageview_cnt == label_cnt:
                raw_labels = list(reversed(mini_batch["label_list"]))
                mini_batch["label_list"] = [
                    raw_labels.pop() if e == "pageview" else 0 for e in mini_batch["event_list"]
                ]
                mini_batch["url_sanity"] = torch.ones(1)
            else:
                mini_batch["label_list"] = [0] * len(mini_batch["event_list"])
                mini_batch["url_sanity"] = torch.zeros(1)
        if event_cnt < 16:
            mini_batch["url_sanity"] = torch.zeros(1)
        label_list = mini_batch["label_list"]

        mini_batch["label_list"] = np.array(label_list, dtype=int)
        mini_batch["label_hots"] = self.url_eye[np.array(label_list, dtype=int)]

        agent = mini_batch["agent_list"]
        mini_batch["agent_list"] = np.array([[1] if a.startswith == "[mobile" else [0] for a in agent])
        return mini_batch

    def time_stamp(self, mini_batch):
        # subsequence method for tracking_time_list
        ts = mini_batch["tracking_date_list"]
        mini_batch["time_delay"] = [t - ts[0] for t in ts]
        return mini_batch

    def relative_size(self, mini_batch):
        # subsequence method to calculate position of click event
        mini_batch["event_list"] = np.array(mini_batch["event_list"])
        events = mini_batch["event_list"]
        location = list(reversed(mini_batch["location_list"]))
        mini_batch["location_list"] = np.array(
            [ast.literal_eval(location.pop()) if e == "click" else [0, 0] for e in events]
        )
        mini_batch["window_list"] = np.array([ast.literal_eval(l) for l in mini_batch["window_list"]])
        mini_batch["page_list"] = np.array([ast.literal_eval(l.replace("null", "0")) for l in mini_batch["page_list"]])
        mini_batch["screen_list"] = np.array([ast.literal_eval(l) for l in mini_batch["screen_list"]])

        return mini_batch

    def purchase(self, mini_batch):
        # mini_batch['purchase'] = torch.tensor([True if max(mini_batch['event_list']) == 3 else False])
        purchase_idx = np.where(mini_batch["event_list"] == "purchase")[0]
        if not purchase_idx:
            mini_batch["purchase"] = torch.tensor([False])
        else:
            mini_batch["purchase"] = torch.tensor([True])
            sorted_cols = [k for k in mini_batch.keys() if (k.endswith("list") and not k.startswith("request"))]
            for c in sorted_cols:
                mini_batch[c] = mini_batch[c][: purchase_idx[0]]

        return mini_batch

    def events(self, mini_batch):
        # subsequence method for event_list
        event_list = mini_batch["event_list"]
        mini_batch["event_hots"] = [self.event_eye[self.event_map[e]] for e in event_list]
        return mini_batch

    def emb(self, mini_batch):
        try:
            emb_vec = torch.cat(
                [
                    torch.stack(mini_batch["event_hots"]),  # length 3 + 1(no-event space)
                    torch.tensor(mini_batch["label_hots"]),  # length 14
                    torch.tensor(mini_batch["location_list"]),  # length 2
                    torch.tensor(mini_batch["window_list"]),  # length 2
                    torch.tensor(mini_batch["page_list"]),  # length 2
                    torch.tensor(mini_batch["screen_list"]),  # length 2
                    torch.tensor(mini_batch["agent_list"]),  # length 1
                ],
                dim=1,
            )  # Total length of feature : 27
        except TypeError:
            emb_vec = torch.cat(
                [
                    torch.stack(mini_batch["event_hots"]),  # length 3 + 1(no-event space)
                    torch.tensor(mini_batch["label_hots"]),  # length 14
                    torch.tensor(mini_batch["location_list"]),  # length 2
                    torch.tensor(mini_batch["location_list"]),  # length 2
                    torch.tensor(mini_batch["location_list"]),  # length 2
                    torch.tensor(mini_batch["location_list"]),  # length 2
                    torch.tensor(mini_batch["location_list"][:, 0:1]),  # length 1
                ],
                dim=1,
            )  # Tota

        canvas = torch.clone(self.canvas)
        t_canvas = torch.ones(len(canvas)) * -1  # represent timestamp for no-event element as -1

        if len(emb_vec) > len(canvas):
            overlap = emb_vec[: len(canvas)]
            t_canvas[: len(overlap)] = torch.tensor(mini_batch["time_delay"][: len(canvas)])
        else:
            canvas[: len(emb_vec), :] = emb_vec
            canvas[len(emb_vec) :, 0] = 1
            t_canvas[: len(emb_vec)] = torch.tensor(mini_batch["time_delay"])
        mini_batch["emb"] = canvas
        mini_batch["t_delay"] = t_canvas
        return mini_batch

    def __call__(self, batch: list[dict], *args, **kwargs):
        result = defaultdict(list)
        for b in batch:
            for logic in self.logics:
                b = self.func_map[logic](b)
            for k, v in b.items():
                result[k].append(v)

        tensor_id = ["emb", "t_delay", "purchase", "url_sanity"]
        return {k: torch.stack(v) if k in tensor_id else v for k, v in result.items()}

        # out = {i: result[i] for i in ['site_id', 'container_id', 'user_id', 'browser_id', 'session_id', 'purchase']}
        # return {k: v for k, v in result.items()}
