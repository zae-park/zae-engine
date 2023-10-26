from typing import Union, Optional, Tuple, List, Any

import numpy as np
import torch
from typeguard import typechecked

from zae_engine.operation import (
    draw_confusion_matrix,
)


EPS = np.finfo(np.float32).eps


def accuracy(
    true: Union[np.ndarray, torch.Tensor],
    predict: Union[np.ndarray, torch.Tensor],
):
    if isinstance(true, torch.Tensor):
        true = true.numpy()
    if isinstance(predict, torch.Tensor):
        predict = predict.numpy()
    assert true.shape == predict.shape, f"Shape unmatched: arg #1 {true.shape} =/= arg #2 {predict.shape}"
    return (true == predict).astype(float).mean()


def rms(recordings):
    return np.sqrt(np.mean(np.square(recordings), axis=-1))


def mse(
    signal1: Union[np.ndarray, torch.Tensor],
    signal2: Union[np.ndarray, torch.Tensor],
):
    assert signal1.shape == signal2.shape, f"Shape unmatched: arg #1 {signal1.shape} =/= arg #2 {signal2.shape}"
    if isinstance(signal1, np.ndarray):
        signal1 = torch.tensor(signal1)
    if isinstance(signal2, np.ndarray):
        signal2 = torch.tensor(signal2)

    return torch.mean(torch.square(signal1 - signal2), dim=-1)


def signal_to_noise(signals, noise):
    """
    Compute signal to noise ratio.
    """
    return 20 * np.log10(rms(signals) / rms(noise) + torch.finfo(torch.float32).eps)


def peak_signal_to_noise(signals, noisy, peak: float = 6.0):
    """
    Compute signal to noise ratio. Assume that peak value is 6[mV].
    """
    return 20 * np.log10(peak / mse(signal1=signals, signal2=noisy))


@typechecked
def miou(
    outputs: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """
    Compute mean IoU for given outputs and labels.
    :param outputs: Shape - [-1, dim]. tensor (or nd-array) of model's outputs.
    :param labels: Shape - [-1, dim]. tensor (or nd-array) of labels.
    :return: mIoU with shape [-1].
    """

    assert (
        "int" in str(outputs.dtype).lower() or "bool" in str(outputs.dtype).lower()
    ), f"outputs array's elements data type must be int or bool type current element type is {outputs.dtype}"

    assert (
        "int" in str(labels.dtype) or "bool" in str(labels.dtype).lower()
    ), f"labels array's elements data type must be int or bool type current element type is {labels.dtype}"

    assert outputs.shape == labels.shape, f"Shape unmatched: arg #1 {outputs.shape} =/= arg #2 {labels.shape}"
    if isinstance(outputs, np.ndarray):
        outputs = torch.tensor(outputs)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    if len(labels.shape) == 1:
        labels = labels.clone().reshape(1, -1)
        outputs = outputs.clone().reshape(1, -1)
    n = len(labels)

    maximum = int(max(outputs.max(), labels.max()))
    iou_ = torch.zeros(n)
    for m in range(maximum):
        intersection = ((outputs == m).int() & (labels == m).int()).float().sum(-1)
        union = ((outputs == m).int() | (labels == m).int()).float().sum(-1)
        iou_ += intersection / (union + torch.finfo(torch.float32).eps)

    return iou_ / maximum


@typechecked
def giou(
    true_onoff: Union[np.ndarray, torch.Tensor, List[Union[int]]],
    pred_onoff: Union[np.ndarray, torch.Tensor, List[Union[int]]],
    iou: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compute mean GIoU and IoU for given outputs and labels.
    :param true_onoff: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param pred_onoff: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param iou: if True, return IoU with GIoU. Default is False.
    :return: GIoU, iou (option) with shape [-1].
    """

    if not isinstance(true_onoff, torch.Tensor):
        true_onoff = torch.tensor(true_onoff)
    if not isinstance(pred_onoff, torch.Tensor):
        pred_onoff = torch.tensor(pred_onoff)

    assert (
        "int" in str(true_onoff.dtype).lower()
    ), f"true_onoff array's elements data type must be int, but receive {true_onoff.dtype}"

    assert (
        "int" in str(pred_onoff.dtype).lower()
    ), f"pred_onoff array's elements data type must be int, but receive {pred_onoff.dtype}"

    if len(true_onoff.shape) == 1:
        true_onoff = true_onoff.clone().unsqueeze(0)
    if len(pred_onoff.shape) == 1:
        pred_onoff = pred_onoff.clone().unsqueeze(0)
    assert (
        true_onoff.shape == pred_onoff.shape
    ), f"Shape unmatched: arg #1 {true_onoff.shape} =/= arg #2 {pred_onoff.shape}"

    true_on, true_off = true_onoff[:, 0], true_onoff[:, 1]
    pred_on, pred_off = pred_onoff[:, 0], pred_onoff[:, 1]
    C_on = torch.min(true_on, pred_on)
    C_off = torch.max(true_off, pred_off)

    eps = +torch.finfo(torch.float32).eps
    C_area = C_off - C_on
    relative_area = C_area - (true_off - true_on)
    union = C_area  # they are same in 1-dimension
    intersection = torch.min(abs(true_on - pred_off), abs(true_off - pred_on))

    IoU = intersection / (union + eps)
    if iou:
        return IoU - abs(relative_area / (C_area + eps)), IoU
    else:
        return IoU - abs(relative_area / (C_area + eps))


def qilv(signal1, signal2, window):
    """
    Quality Index based on Local Variance (QILV) - Santiago Aja Fernandez (santi @ bwh.harhard.edu)
    Ref - "Image quality assessment based on local variance", EMBC 2006
    ------------------------------------------------------------------
    Calculate a global compatibility measure between two images, based on their local variance distribution.
    TODO: Seems not valid.

    INPUT:
        signal1, signal2: Two signals for calculation.
        window_size: size of window to define range of local.
        window_type: Type of window to define coefficient of window.
    OUTPUT:
        qi: Quality index. bounded into [0, 1].
    USAGE:
        qi = qilv1D(sig1, sig2, 7)
    """
    window_ = window / np.sum(window)  # normalized

    # Local statistics
    l_means1 = np.convolve(signal1, window_, "valid")
    l_means2 = np.convolve(signal2, window_, "valid")
    l_vars1 = np.convolve(signal1**2, window_, "valid") - l_means1**2
    l_vars2 = np.convolve(signal2**2, window_, "valid") - l_means2**2

    # Global statistics
    mean_l_vars1, mean_l_vars2 = np.mean(l_vars1), np.mean(l_vars2)
    std_l_vars1, std_l_vars2 = np.std(l_vars1), np.std(l_vars2)
    covar_l_vars = np.mean((l_vars1 - mean_l_vars1) * (l_vars2 - mean_l_vars2))

    index1 = (2 * mean_l_vars1 * mean_l_vars2) / (
        mean_l_vars1**2 + mean_l_vars2**2 + torch.finfo(torch.float32).eps
    )
    index2 = (2 * std_l_vars1 * std_l_vars2) / (std_l_vars1**2 + std_l_vars2**2 + torch.finfo(torch.float32).eps)
    index3 = covar_l_vars / (std_l_vars1 * std_l_vars2 + torch.finfo(torch.float32).eps)

    return index1 * index2 * index3


def fbeta(*args, beta: float, num_classes: int, average: str = "micro"):
    """
    Compute f-beta score using given confusion matrix (args#1 with asterisk).
    If the first argument is a tuple of length 2, i.e. true and prediction, then compute confusion matrix first.
    Support two average methods to calculate precision or recall, i.e. micro- and macro-.

    :param args:
        Confusion matrix (or true and prediction)
    :param beta: float
    :param num_classes: int
    :param average: str
        If 'micro', precision and recall are derived using TP and FP for all classes.
        If 'macro', precision and recall are derived using precision and recall for each class.
    """
    eps = torch.finfo(torch.float32).eps
    if len(args) == 2:
        conf = draw_confusion_matrix(*args, num_classes=num_classes)
    else:
        conf = args[0]

    tp_set = conf.diagonal()
    row = conf.sum(1)
    col = conf.sum(0)

    if average == "micro":
        micro_tp = tp_set.sum()
        micro_fn = (row - tp_set).sum()
        micro_fp = (col - tp_set).sum()

        recall = micro_tp / (micro_tp + micro_fn + eps)
        precision = micro_tp / (micro_tp + micro_fp + eps)

        micro_f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + eps)
        return micro_f1

    elif average == "macro":
        macro_f1 = 0
        for tp, r, c in zip(tp_set, row, col):
            precision = tp / (c + eps)
            recall = tp / (r + eps)
            f1 = (1 + beta**2) * recall * precision / ((beta**2) * precision + recall + eps)
            macro_f1 += f1

        return macro_f1 / num_classes


def iec_60601(true: dict, predict: dict, data_length: int, criteria: str) -> Tuple[float, float]:
    """
    Calculate sensitivity (se) & positive predictive value (pp) of criteria with given true & predict annotation.

    pp = Intersection / algorithm_Interval
    se = Intersection / true_interval

    from IEC 60601-2-47[https://webstore.iec.ch/publication/2666]

    :param true: dict
    :param predict: dict
    :param data_length: int
    :param criteria: str
    :return: Tuple[float, float]
    """

    assert (
        (t1 := type(true)) == (t2 := type(predict)) == dict
    ), f"Expect type of given arguments are Dictionary, but receive {t1} and {t2}"

    def get_set(anno) -> set:
        """
        Calculate sample set in annotation with criteria.
        :param anno: Dictionary
        :return: set
        """
        ON, sample_set = False, set()

        assert len(samples := anno["sample"]) == len(aux := anno["rhythm"]), f"Lengths of keys must be same."
        assert samples == sorted(samples), f'Key "sample" must be sorted.'

        for i, a in zip(samples, aux):
            if criteria in a:
                if not ON:
                    ON = i
            else:
                if ON:
                    sample_set = sample_set.union(list(range(ON, i)))
                    ON = False
        if ON:
            sample_set = sample_set.union(list(range(ON, data_length)))
        return sample_set

    try:
        true_set, pred_set = get_set(true), get_set(predict)
    except KeyError:
        raise KeyError('Given arguments must have following keys, ["sample", "rhythm"].')
    except AssertionError as e:
        raise e
    else:
        inter = true_set.intersection(pred_set)
        pp, se = len(inter) / (len(pred_set) + EPS), len(inter) / (len(true_set) + EPS)
        return pp, se


def cpsc2021(true: list, predict: list, margin: int = 3) -> float:
    """
    Calculate modified CPSC score with given true onoff list & predict onoff list.
    Find predict on point and off point in true onoff list with margin.

    score = correct_onoff_predict / max(len(pred_onoff_set), len(true_onoff_set))
    from cpsc2021[http://2021.icbeb.org/CPSC2021]


    :param true: list
    :param predict: list
    :param margin: int
    :return: float
    """

    assert len(true) % 2 == len(predict) % 2 == 0, f"Expect the length of each input argument to be even."

    on, off = [], []
    for i, t in enumerate(true):
        set_to = off if i % 2 else on
        set_to += list(range(t - margin, t + margin))
    on, off = set(on), set(off)

    score = 0
    for i, p in enumerate(predict):
        find_set = off if i % 2 else on
        if p in find_set:
            score += 1
    score /= 2
    n_episode = max(len(true) // 2, len(predict) // 2)
    return score / n_episode
