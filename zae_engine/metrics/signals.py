import math
from typing import Union

import numpy as np
import torch

from ..utils.decorators import np2torch, torch2np


EPS = torch.finfo(torch.float32).eps


def rms(signal: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute the root mean square (RMS) of a signal.

    This function calculates the RMS value of the input signal, which is a measure of the magnitude of the signal.

    Parameters
    ----------
    signal : Union[np.ndarray, torch.Tensor]
        The input signal, either as a numpy array or a torch tensor.

    Returns
    -------
    float
        The RMS value of the signal.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> rms(signal)
    3.3166247903554
    >>> signal = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    >>> rms(signal)
    3.3166247901916504
    """
    return ((signal**2).mean() ** 0.5).item()


@np2torch(torch.float)
def mse(signal1: Union[np.ndarray, torch.Tensor], signal2: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute the mean squared error (MSE) between two signals.

    This function calculates the MSE between two input signals, which is a measure of the average squared difference between the signals.

    Parameters
    ----------
    signal1 : Union[np.ndarray, torch.Tensor]
        The first input signal, either as a numpy array or a torch tensor.
    signal2 : Union[np.ndarray, torch.Tensor]
        The second input signal, either as a numpy array or a torch tensor.

    Returns
    -------
    float
        The MSE value of the signal.

    Examples
    --------
    >>> signal1 = np.array([1, 2, 3, 4, 5])
    >>> signal2 = np.array([1, 2, 3, 4, 6])
    >>> mse(signal1, signal2)
    0.20000000298023224
    >>> signal1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    >>> signal2 = torch.tensor([1, 2, 3, 4, 6], dtype=torch.float)
    >>> mse(signal1, signal2)
    0.20000000298023224
    """
    return ((signal1 - signal2) ** 2).mean().item()


@np2torch(torch.float)
def signal_to_noise(signal: Union[torch.Tensor, np.ndarray], noise: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the signal-to-noise ratio (SNR).

    This function calculates the SNR, which is a measure of the ratio of the power of a signal to the power of background noise.

    Parameters
    ----------
    signal : Union[torch.Tensor, np.ndarray]
        The input signal, either as a torch tensor or a numpy array.
    noise : Union[torch.Tensor, np.ndarray]
        The noise signal, either as a torch tensor or a numpy array.

    Returns
    -------
    float
        The SNR value in decibels (dB).

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> noise = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> signal_to_noise(signal, noise)
    20.0
    >>> signal = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    >>> noise = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float)
    >>> signal_to_noise(signal, noise)
    20.0
    """
    signal_eff, noise_eff = rms(signal=signal), rms(signal=noise)
    db = 20 * math.log10(signal_eff / (noise_eff + EPS))
    return db


@np2torch(torch.float)
def peak_signal_to_noise(
    signal: Union[torch.Tensor, np.ndarray],
    noise: Union[torch.Tensor, np.ndarray],
    peak: Union[bool, int, float] = False,
) -> float:
    """
    Compute the peak signal-to-noise ratio (PSNR).

    This function calculates the PSNR, which is a measure of the ratio of the peak signal power to the power of background noise.
    If peak is not given, the maximum value in the signal is used as the peak value.

    Parameters
    ----------
    signal : Union[torch.Tensor, np.ndarray]
        The input signal, either as a torch tensor or a numpy array.
    noise : Union[torch.Tensor, np.ndarray]
        The noise signal, either as a torch tensor or a numpy array.
    peak : Union[bool, int, float], optional
        The peak value to be used in the calculation. If not provided, the maximum value in the signal is used.

    Returns
    -------
    float
        The PSNR value in decibels (dB).

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> noise = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> peak_signal_to_noise(signal, noise)
    24.0824
    >>> signal = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    >>> noise = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float)
    >>> peak_signal_to_noise(signal, noise)
    24.0824
    >>> peak_signal_to_noise(signal, noise, peak=5)
    24.0824
    """
    if not peak:
        peak = signal.max()

    mse_value = mse(signal1=signal, signal2=noise)
    db = 20 * math.log10(peak / math.sqrt(mse_value + EPS))
    return db


@torch2np(np.float32)
def qilv(
    signal1: Union[torch.Tensor, np.ndarray],
    signal2: Union[torch.Tensor, np.ndarray],
    window: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Calculate the Quality Index based on Local Variance (QILV) for two signals.

    This function computes a global compatibility metric between two signals based on their local variance distribution.

    Parameters
    ----------
    signal1 : Union[torch.Tensor, np.ndarray]
        The first input signal.
    signal2 : Union[torch.Tensor, np.ndarray]
        The second input signal.
    window : Union[torch.Tensor, np.ndarray]
        The window used to define the local region for variance calculation.

    Returns
    -------
    float
        The QILV value, bounded between [0, 1].

    References
    ----------
    .. [1] Santiago Aja-Fernández et al. "Image quality assessment based on local variance", EMBC 2006.
           https://ieeexplore.ieee.org/document/4481769

    Examples
    --------
    >>> signal1 = np.array([1, 2, 3, 4, 5])
    >>> signal2 = np.array([1, 2, 3, 4, 6])
    >>> window = np.ones(3)
    >>> qilv(signal1, signal2, window)
    0.9948761003700519
    """
    # Normalize the window
    window_ = window / np.sum(window)

    # Local statistics
    l_means1 = np.convolve(signal1, window_, "valid")
    l_means2 = np.convolve(signal2, window_, "valid")
    l_vars1 = np.convolve(signal1**2, window_, "valid") - l_means1**2
    l_vars2 = np.convolve(signal2**2, window_, "valid") - l_means2**2

    # Global statistics
    mean_l_vars1, mean_l_vars2 = np.mean(l_vars1), np.mean(l_vars2)
    std_l_vars1, std_l_vars2 = np.std(l_vars1), np.std(l_vars2)
    covar_l_vars = np.mean((l_vars1 - mean_l_vars1) * (l_vars2 - mean_l_vars2))

    # Calculate indices
    index1 = (2 * mean_l_vars1 * mean_l_vars2) / (mean_l_vars1**2 + mean_l_vars2**2 + np.finfo(np.float32).eps)
    index2 = (2 * std_l_vars1 * std_l_vars2) / (std_l_vars1**2 + std_l_vars2**2 + np.finfo(np.float32).eps)
    index3 = covar_l_vars / (std_l_vars1 * std_l_vars2 + np.finfo(np.float32).eps)

    qilv_value = index1 * index2 * index3
    return max(0.0, min(1.0, qilv_value))


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
