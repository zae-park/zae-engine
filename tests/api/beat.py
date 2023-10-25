from collections import defaultdict
from typing import Union, Tuple, List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from zae_engine import models, trainer
from zae_engine.data.collate import BeatCollateSeq as Col
from zae_engine.operation import label_to_onoff, sanity_check, onoff_to_label


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
    model = models.beat_segmentation(True)
    trainer1 = Trainer_stg1(model=model, device=device, mode="test")
    prediction_stg1 = np.concatenate(trainer1.inference(inference_loader)).argmax(1)

    # postprocessing
    cursor, fin_dict = 0, defaultdict(list)
    for col in inference_loader.__iter__():
        n = col["x"].__len__()
        fin_dict["x"].append(col["raw"])
        pred_onoff = label_to_onoff(prediction_stg1[cursor : cursor + n], sense=5)
        pred_onoff = [sanity_check(oo) for oo in pred_onoff]
        pred_onoff = recursive_peak_merge(pred_onoff, len(pred_onoff) // 2, 250, 2)[0]
        try:
            if not pred_onoff:
                raise np.AxisError(0)
            fin_dict["onoff"].append(pred_onoff[pred_onoff.max(1) < len(col["raw"])])
        except np.AxisError:
            fin_dict["onoff"].append(None)
        except ValueError:
            fin_dict["onoff"].append(pred_onoff[pred_onoff.max(1) < len(col["raw"])])
        fin_dict["y"].append(onoff_to_label(np.array(pred_onoff), length=len(col["raw"])).tolist())
        fin_dict["fn"].append(col["fn"])
        cursor += n

    # --------------------------------- Load data @ stage 2 --------------------------------- #
    inference_dataset2 = ECG_dataset(
        x=np.stack(fin_dict["x"]), y=np.stack(fin_dict["y"]), fn=fin_dict["fn"], onoff=fin_dict["onoff"]
    )
    inference_loader2 = DataLoader(
        inference_dataset2, batch_size=1, shuffle=False, collate_fn=Col(sequence=["r_reg"], zit=False, resamp=64).wrap()
    )

    # --------------------------------- Inference & Postprocess @ stage 2 --------------------------------- #
    model = models.rpeak_regression(True)
    trainer2 = Trainer_stg2(model=model, device=device, mode="test")
    prediction_stg2 = torch.cat(trainer2.inference(inference_loader2)).squeeze().numpy()

    # postprocessing
    cursor = 0
    for i, onoff in enumerate(inference_dataset2.onoff):
        if onoff is None:
            continue
        n = onoff.__len__()
        r_loc = prediction_stg2[cursor : cursor + n]
        r_idx = onoff[:, 0] + (onoff[:, 1] - onoff[:, 0]) * r_loc
        fin_onoff = np.concatenate((onoff, r_idx.reshape(-1, 1)), axis=1).astype(int)
        fin_dict["fin_onoff"].append(fin_onoff[fin_onoff[:, -1] < len(fin_dict["x"][i])])
        cursor += n

    if fin_dict["fin_onoff"]:
        return [onoff.tolist() for onoff in fin_dict["fin_onoff"]][0]
    else:
        return []


def recursive_peak_merge(qrs_indexes, half_index, sampling_rate, overlapped_sec):
    """merge overlapped 10sec data.
    Args:
         qrs_indexes: qrs info, N x B x (on off rp cls). N is # of 10 sec data, B is # of beats in a data.
         half_index: half # of 10 sec data.
         sampling_rate: # of pnts in a second.
         overlapped_sec: data is merged according to this param.
    """

    n_overlapped_samples = sampling_rate * overlapped_sec  # 500
    merged_qrs_indexes = np.empty((0, 3), int)  # dummy

    if len(qrs_indexes) == 1:
        return np.array(qrs_indexes)  # data가 1개라 merge 할 게 없음.
    elif len(qrs_indexes) == 2:
        if len(qrs_indexes[0]) > 0:
            merged_qrs_indexes = np.array(qrs_indexes[0])  # 1번째 data에 뭐가 있는 경우
        if len(qrs_indexes[1]) == 0:
            return np.expand_dims(merged_qrs_indexes, axis=0)  # 2번째 data가 빈 경우 땡큐

        shift = int(half_index * (sampling_rate * 10 - n_overlapped_samples))  # half_index는 아마 1?
        shifted_peak_indexes = np.array(qrs_indexes[1])  # 2번째 data
        shifted_peak_indexes[:, 0:2] = shifted_peak_indexes[:, 0:2] + shift

        # 누적된 신호에서 오버랩 후보 영역을 찾는다
        overlapped_pos = np.where(merged_qrs_indexes[:, 1] >= shift - 10)[0]
        overlapped_indexes = merged_qrs_indexes[overlapped_pos]
        # 현재 10초 신호에서 overlap 가능성이 있는 앞부분만 떼어내기
        shifted_overlapped_pos = np.where(np.array(qrs_indexes[1])[:, 0] < n_overlapped_samples)[0]
        shifted_overlapped_indexes = shifted_peak_indexes[shifted_overlapped_pos]

        if len(overlapped_indexes) == 0 or len(shifted_overlapped_indexes) == 0:  # overlap이 없으면 그냥 합치기
            if len(merged_qrs_indexes) > 0 and len(shifted_peak_indexes) > 0:
                merged_qrs_indexes = np.concatenate((merged_qrs_indexes, shifted_peak_indexes), axis=0)
            elif len(shifted_peak_indexes) > 0:
                merged_qrs_indexes = shifted_peak_indexes
        else:  # overlap이 있는 경우
            # 겹치는 qrs 찾기
            # qrs 중심 거리가 기존에 누적된 qrs 중심 거리와 30 이내라면 중복 qrs로 취급
            duplicated = [False] * len(shifted_overlapped_indexes)
            for j, shifted_index in enumerate(shifted_overlapped_indexes):
                shifted_qrs_center = shifted_index[0] + (shifted_index[1] - shifted_index[0]) / 2
                for k, overlapped_index in enumerate(overlapped_indexes):
                    overlapped_qrs_center = overlapped_index[0] + (overlapped_index[1] - overlapped_index[0]) / 2
                    if abs(shifted_qrs_center - overlapped_qrs_center) < 30:
                        duplicated[j] = True
                        break
                    if not (overlapped_index[1] < shifted_index[0] or shifted_index[1] < overlapped_index[0]):
                        duplicated[j] = True
                        break
            # overlap 구간에서 중복되지 않는 모든 qrs 붙이기
            overlapped = np.concatenate(
                (overlapped_indexes, shifted_overlapped_indexes[np.where(np.array(duplicated) == False)]), axis=0
            )
            overlapped = sorted(overlapped, key=lambda x: x[0])

            merged_qrs_indexes = np.concatenate(
                (
                    merged_qrs_indexes[: overlapped_pos[0]],
                    overlapped,
                    shifted_peak_indexes[shifted_overlapped_pos[-1] + 1 :],
                ),
                axis=0,
            )

        return np.expand_dims(merged_qrs_indexes, axis=0)
    else:
        # half... recursive
        n = len(qrs_indexes) // 2
        m1 = recursive_peak_merge(qrs_indexes[:n], n // 2, sampling_rate, overlapped_sec)
        m2 = recursive_peak_merge(qrs_indexes[n:], (len(qrs_indexes) - n) // 2, sampling_rate, overlapped_sec)
        m1 = m1.tolist()
        m2 = m2.tolist()
        m = m1 + m2
        return recursive_peak_merge(m, n, sampling_rate, overlapped_sec)


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
        print(f"\t # of {self.mode} data: %d" % len(self.x))

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


class Trainer_stg1(trainer.Trainer):
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


class Trainer_stg2(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(Trainer_stg2, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 128

    def train_step(self, batch: dict):
        pass

    def test_step(self, batch: dict):
        if batch is None:
            return {"loss": 0, "output": torch.Tensor([]), "output_onoff": np.array([])}
        if isinstance((w := batch["w"]), torch.Tensor) and isinstance((onoff := batch["onoff"]), list):
            mini_w = w.split(self.mini_batch_size, dim=0)
            pred = torch.cat([self.model(m_w) for m_w in mini_w])
            return {"loss": 0, "output": self.model.clipper(pred), "output_onoff": onoff}
        else:
            return {"loss": 0, "output": torch.Tensor([]), "output_onoff": np.array([])}
