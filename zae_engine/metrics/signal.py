from typing import Union, Optional, List

import numpy as np
import torch
from typeguard import typechecked

from ..operation import draw_confusion_matrix
from ..utils.deco import np2torch


EPS = torch.finfo(torch.float32).eps


def rms(signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return (signal ** 2).mean() ** 0.5


@np2torch(torch.float)
def mse(signal1: Union[np.ndarray, torch.Tensor], signal2:Union[np.ndarray, torch.Tensor]):
    return ((signal1 - signal2) ** 2).mean()


@np2torch(torch.float)
def signal_to_noise(signal: Union[torch.Tensor | np.ndarray], noise: Union[torch.Tensor | np.ndarray]):
    """
    Compute signal-to-noise ratio.
    """
    signal_eff, noise_eff = rms(signal=signal), rms(signal=noise)
    db = 20 * torch.log10(signal_eff / (noise_eff + EPS))

    return db


@np2torch(torch.float)
def peak_signal_to_noise(signal, noise, peak: Union[bool | int | float] = False):
    """
    Compute peak-signal-to-noise ratio.
    If peak is not given, use maximum value is signal instead.
    """
    if not peak:
        peak = signal.max()
    db = 20 * torch.log10(peak / mse(signal1=signal, signal2=noise))
    return db


# def qilv(signal1, signal2, window):
#     """
#     Quality Index based on Local Variance (QILV) - Santiago Aja Fernandez (santi @ bwh.harhard.edu)
#     Ref - "Image quality assessment based on local variance", EMBC 2006
#     ------------------------------------------------------------------
#     Calculate a global compatibility metrics between two images, based on their local variance distribution.
#     TODO: Seems not valid.
#
#     INPUT:
#         signal1, signal2: Two signals for calculation.
#         window_size: size of window to define range of local.
#         window_type: Type of window to define coefficient of window.
#     OUTPUT:
#         qi: Quality index. bounded into [0, 1].
#     USAGE:
#         qi = qilv1D(sig1, sig2, 7)
#     """
#     window_ = window / np.sum(window)  # normalized
#
#     # Local statistics
#     l_means1 = np.convolve(signal1, window_, "valid")
#     l_means2 = np.convolve(signal2, window_, "valid")
#     l_vars1 = np.convolve(signal1**2, window_, "valid") - l_means1**2
#     l_vars2 = np.convolve(signal2**2, window_, "valid") - l_means2**2
#
#     # Global statistics
#     mean_l_vars1, mean_l_vars2 = np.mean(l_vars1), np.mean(l_vars2)
#     std_l_vars1, std_l_vars2 = np.std(l_vars1), np.std(l_vars2)
#     covar_l_vars = np.mean((l_vars1 - mean_l_vars1) * (l_vars2 - mean_l_vars2))
#
#     index1 = (2 * mean_l_vars1 * mean_l_vars2) / (mean_l_vars1**2 + mean_l_vars2**2 + torch.finfo(torch.float32).eps)
#     index2 = (2 * std_l_vars1 * std_l_vars2) / (std_l_vars1**2 + std_l_vars2**2 + torch.finfo(torch.float32).eps)
#     index3 = covar_l_vars / (std_l_vars1 * std_l_vars2 + torch.finfo(torch.float32).eps)
#
#     return index1 * index2 * index3
#
#

# def iec_60601(true: dict, predict: dict, data_length: int, criteria: str) -> Tuple[float, float]:
#     """
#     Calculate sensitivity (se) & positive predictive value (pp) of criteria with given true & predict annotation.
#
#     pp = Intersection / algorithm_Interval
#     se = Intersection / true_interval
#
#     from IEC 60601-2-47[https://webstore.iec.ch/publication/2666]
#
#     :param true: dict
#     :param predict: dict
#     :param data_length: int
#     :param criteria: str
#     :return: Tuple[float, float]
#     """
#
#     assert (
#         (t1 := type(true)) == (t2 := type(predict)) == dict
#     ), f"Expect type of given arguments are Dictionary, but receive {t1} and {t2}"
#
#     def get_set(anno) -> set:
#         """
#         Calculate sample set in annotation with criteria.
#         :param anno: Dictionary
#         :return: set
#         """
#         ON, sample_set = False, set()
#
#         assert len(samples := anno["sample"]) == len(aux := anno["rhythm"]), f"Lengths of keys must be same."
#         assert samples == sorted(samples), f'Key "sample" must be sorted.'
#
#         for i, a in zip(samples, aux):
#             if criteria in a:
#                 if not ON:
#                     ON = i
#             else:
#                 if ON:
#                     sample_set = sample_set.union(list(range(ON, i)))
#                     ON = False
#         if ON:
#             sample_set = sample_set.union(list(range(ON, data_length)))
#         return sample_set
#
#     try:
#         true_set, pred_set = get_set(true), get_set(predict)
#     except KeyError:
#         raise KeyError('Given arguments must have following keys, ["sample", "rhythm"].')
#     except AssertionError as e:
#         raise e
#     else:
#         inter = true_set.intersection(pred_set)
#         pp, se = len(inter) / (len(pred_set) + EPS_N), len(inter) / (len(true_set) + EPS_N)
#         return pp, se
#
#
# def cpsc2021(true: list, predict: list, margin: int = 3) -> float:
#     """
#     Calculate modified CPSC score with given true onoff list & predict onoff list.
#     Find predict on point and off point in true onoff list with margin.
#
#     score = correct_onoff_predict / max(len(pred_onoff_set), len(true_onoff_set))
#     from cpsc2021[http://2021.icbeb.org/CPSC2021]
#
#
#     :param true: list
#     :param predict: list
#     :param margin: int
#     :return: float
#     """
#
#     assert len(true) % 2 == len(predict) % 2 == 0, f"Expect the length of each input argument to be even."
#
#     on, off = [], []
#     for i, t in enumerate(true):
#         set_to = off if i % 2 else on
#         set_to += list(range(t - margin, t + margin))
#     on, off = set(on), set(off)
#
#     score = 0
#     for i, p in enumerate(predict):
#         find_set = off if i % 2 else on
#         if p in find_set:
#             score += 1
#     score /= 2
#     n_episode = max(len(true) // 2, len(predict) // 2)
#     return score / n_episode
