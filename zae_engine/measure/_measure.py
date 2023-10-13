from typing import Union, Optional, Tuple, List, Any

import numpy as np
import torch
import wfdb
from typeguard import typechecked

from zae_engine.operation import label_to_onoff, onoff_to_label, find_nearest, sanity_check,\
    draw_confusion_matrix, print_confusion_matrix


EPS = np.finfo(np.float32).eps

class BijectiveMetric:
    def __init__(self,
                 prediction: Union[np.ndarray, torch.Tensor],
                 label: Union[np.ndarray, torch.Tensor], num_class: int, for_beat: bool = True, th_onoff: int = 2):
        """
        Compute bijective confusion matrix of given sequences.
        The surjective operation projects every onoff in prediction onto label and check IoU.
        The injective projection is vice versa.

        :param prediction: Sequence of prediction.
        :param label: int. Sequence of label.
        """
        self.num_class = num_class
        self.eps = torch.finfo(torch.float32).eps

        assert prediction.shape == label.shape, f'Unmatched shape error. {prediction.shape} =/= {label.shape}'
        assert len(label.shape) == 1, f'Unexpected shape error. Expect 1-D array but receive {label.shape}.'

        self.pred_onoff = label_to_onoff(prediction, outside_idx=for_beat, sense=th_onoff)
        self.pred_onoff = torch.tensor(sanity_check(self.pred_onoff, incomplete_only=True), dtype=torch.int)
        self.pred_array = onoff_to_label(self.pred_onoff, length=len(prediction))     # w/o incomplete

        self.label_onoff = label_to_onoff(label, outside_idx=for_beat, sense=th_onoff)
        self.label_onoff = torch.tensor(sanity_check(self.label_onoff, incomplete_only=True), dtype=torch.int)
        self.label_array = onoff_to_label(self.label_onoff, length=len(prediction))   # w/o incomplete

        self.bijective_onoff, self.injective_onoff, self.surjective_onoff = self.onoff_pairing()

        # Injective: pred --projection-> label
        self.injective_mat = self.map_and_confusion(self.pred_onoff, self.label_array).transpose()
        self.injective_count = self.injective_mat.sum()

        # Surjective: pred <-projection-- label
        self.surjective_mat = self.map_and_confusion(self.label_onoff, self.pred_array)
        self.surjective_count = self.surjective_mat.sum()

        # Bijective: using onoff_pair
        self.bijective_mat = self.bijective_confusion()
        self.bijective_count = self.bijective_mat.sum()

        self.bijective_f1 = fbeta(self.bijective_mat[1:, 1:], beta=1, num_classes=num_class)
        self.injective_f1 = fbeta(self.injective_mat[1:, 1:], beta=1, num_classes=num_class)
        self.surjective_f1 = fbeta(self.surjective_mat[1:, 1:], beta=1, num_classes=num_class)

        self.bijective_acc = self.bijective_mat[1:, 1:].trace() / (self.bijective_mat[1:, 1:].sum() + self.eps)
        self.injective_acc = self.injective_mat[1:, 1:].trace() / (self.injective_mat[1:, 1:].sum() + self.eps)
        self.surjective_acc = self.surjective_mat[1:, 1:].trace() / (self.surjective_mat[1:, 1:].sum() + self.eps)

    def onoff_pairing(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        injective_onoff_pair, surjective_onoff_pair = [], []

        if len(self.label_onoff.shape) == 2:
            for i_p, p_oo in enumerate(self.pred_onoff):
                p_oo = p_oo.tolist()
                i_nearest, v_nearest = find_nearest(self.label_onoff[:, 0], int(p_oo[0]))   # find nearest onoff
                l_oo = self.label_onoff[i_nearest].tolist()
                iou = giou(p_oo[:-1], l_oo[:-1])[-1]
                if iou > 0.5:
                    injective_onoff_pair.append([i_p, i_nearest, p_oo[-1], l_oo[-1]])

        if len(self.pred_onoff.shape) == 2:
            for i_l, l_oo in enumerate(self.label_onoff):
                l_oo = l_oo.tolist()
                i_nearest, v_nearest = find_nearest(self.pred_onoff[:, 0], int(l_oo[0]))    # find nearest onoff
                p_oo = self.pred_onoff[i_nearest].tolist()
                iou = giou(l_oo[:-1], p_oo[:-1])[-1]
                if iou > 0.5:
                    surjective_onoff_pair.append([i_nearest, i_l, p_oo[-1], l_oo[-1]])

        bijective_onoff_pair = []
        for inj in injective_onoff_pair:
            if inj in surjective_onoff_pair:
                bijective_onoff_pair.append(inj)

        injective_onoff_pair = torch.tensor(injective_onoff_pair)
        surjective_onoff_pair = torch.tensor(surjective_onoff_pair)
        return torch.tensor(bijective_onoff_pair), injective_onoff_pair, surjective_onoff_pair

    def map_and_confusion(self, x_onoff: Union[torch.Tensor, np.ndarray], y_array: Union[torch.Tensor, np.ndarray]):
        if isinstance(x_onoff, np.ndarray): x_onoff = torch.tensor(x_onoff, dtype=torch.int)
        if isinstance(y_array, np.ndarray): y_array = torch.tensor(y_array, dtype=torch.int)

        confusion_mat = np.zeros((self.num_class, self.num_class), dtype=np.int32)
        for i_onoff, (start, end, x) in enumerate(x_onoff):
            y = int(y_array[start:end+1].mode().values)
            confusion_mat[x, y] += 1
        return confusion_mat

    def bijective_confusion(self):
        confusion_mat = self.surjective_mat + self.injective_mat
        confusion_mat[1:, 1:] = 0
        for _, _, p, l in self.bijective_onoff:
            confusion_mat[l, p] += 1
        return confusion_mat

    def summary(self, class_name: Union[Tuple, List] = None):
        print("\t\t# of samples in bijective confusion mat -> Bi: {bi} / Inj: {inj} / Sur: {sur}"
              .format(bi=self.bijective_count, inj=self.injective_count, sur=self.surjective_count))
        print("\t\tF-beta score from bijective metric -> Bi: {bi:.2f}% / Inj: {inj:.2f}% / Sur: {sur:.2f}%"
              .format(bi=100 * self.bijective_f1, inj=100 * self.injective_f1, sur=100 * self.surjective_f1))
        print("\t\tAccuracy from bijective metric -> Bi: {bi:.2f}% / Inj: {inj:.2f}% / Sur: {sur:.2f}%"
              .format(bi=100 * self.bijective_acc, inj=100 * self.injective_acc, sur=100 * self.surjective_acc))

        print_confusion_matrix(self.bijective_confusion(), class_name=class_name)

        num_catch = self.bijective_mat[1:, 1:].sum()
        print('Beat Acc : %d / %d -> %.2f%%' %
              (num_catch, self.bijective_count, 100 * num_catch / self.bijective_count + self.eps))
        print()


def accuracy(true: Union[np.ndarray, torch.Tensor], predict: Union[np.ndarray, torch.Tensor]):
    if isinstance(true, torch.Tensor): true = true.numpy()
    if isinstance(predict, torch.Tensor): predict = predict.numpy()
    assert true.shape == predict.shape, f'Shape unmatched: arg #1 {true.shape} =/= arg #2 {predict.shape}'
    return (true == predict).astype(float).mean()


def rms(recordings):
    return np.sqrt(np.mean(np.square(recordings), axis=-1))


def mse(signal1: Union[np.ndarray, torch.Tensor], signal2: Union[np.ndarray, torch.Tensor]):
    assert signal1.shape == signal2.shape, f'Shape unmatched: arg #1 {signal1.shape} =/= arg #2 {signal2.shape}'
    if isinstance(signal1, np.ndarray): signal1 = torch.tensor(signal1)
    if isinstance(signal2, np.ndarray): signal2 = torch.tensor(signal2)

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
def miou(outputs: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Compute mean IoU for given outputs and labels.
    :param outputs: Shape - [-1, dim]. tensor (or nd-array) of model's outputs.
    :param labels: Shape - [-1, dim]. tensor (or nd-array) of labels.
    :return: mIoU with shape [-1].
    """

    assert 'int' in str(outputs.dtype).lower() or 'bool' in str(outputs.dtype).lower(), \
        f'outputs array\'s elements data type must be int or bool type current element type is {outputs.dtype}'

    assert 'int' in str(labels.dtype) or 'bool' in str(labels.dtype).lower(), \
        f'labels array\'s elements data type must be int or bool type current element type is {labels.dtype}'

    assert outputs.shape == labels.shape, f'Shape unmatched: arg #1 {outputs.shape} =/= arg #2 {labels.shape}'
    if isinstance(outputs, np.ndarray): outputs = torch.tensor(outputs)
    if isinstance(labels, np.ndarray): labels = torch.tensor(labels)
    if len(labels.shape) == 1:
        labels = labels.clone().reshape(1, -1)
        outputs = outputs.clone().reshape(1, -1)
    n = len(labels)

    maximum = int(max(outputs.max(), labels.max()))
    iou_ = torch.zeros(n)
    for m in range(maximum):
        intersection = ((outputs == m).int() & (labels == m).int()).float().sum(-1)
        union = ((outputs == m).int() | (labels == m).int()).float().sum(-1)
        iou_ += (intersection / (union + torch.finfo(torch.float32).eps))

    return iou_ / maximum


@typechecked
def giou(true_onoff: Union[np.ndarray, torch.Tensor, List[Union[int]]],
         pred_onoff: Union[np.ndarray, torch.Tensor, List[Union[int]]],
         iou: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compute mean GIoU and IoU for given outputs and labels.
    :param true_onoff: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param pred_onoff: Shape - [-1, 2].
    tensor (or nd-array) of on-off pairs. Each on-off pair corresponds to bounding box in object detection.
    :param iou: if True, return IoU with GIoU. Default is False.
    :return: GIoU, iou (option) with shape [-1].
    """

    if not isinstance(true_onoff, torch.Tensor): true_onoff = torch.tensor(true_onoff)
    if not isinstance(pred_onoff, torch.Tensor): pred_onoff = torch.tensor(pred_onoff)

    assert 'int' in str(true_onoff.dtype).lower(), \
        f'true_onoff array\'s elements data type must be int, but receive {true_onoff.dtype}'

    assert 'int' in str(pred_onoff.dtype).lower(), \
        f'pred_onoff array\'s elements data type must be int, but receive {pred_onoff.dtype}'

    if len(true_onoff.shape) == 1: true_onoff = true_onoff.clone().unsqueeze(0)
    if len(pred_onoff.shape) == 1: pred_onoff = pred_onoff.clone().unsqueeze(0)
    assert true_onoff.shape == pred_onoff.shape, \
        f'Shape unmatched: arg #1 {true_onoff.shape} =/= arg #2 {pred_onoff.shape}'

    true_on, true_off = true_onoff[:, 0], true_onoff[:, 1]
    pred_on, pred_off = pred_onoff[:, 0], pred_onoff[:, 1]
    C_on = torch.min(true_on, pred_on)
    C_off = torch.max(true_off, pred_off)

    eps = + torch.finfo(torch.float32).eps
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
    l_means1 = np.convolve(signal1, window_, 'valid')
    l_means2 = np.convolve(signal2, window_, 'valid')
    l_vars1 = np.convolve(signal1 ** 2, window_, 'valid') - l_means1 ** 2
    l_vars2 = np.convolve(signal2 ** 2, window_, 'valid') - l_means2 ** 2

    # Global statistics
    mean_l_vars1, mean_l_vars2 = np.mean(l_vars1), np.mean(l_vars2)
    std_l_vars1, std_l_vars2 = np.std(l_vars1), np.std(l_vars2)
    covar_l_vars = np.mean((l_vars1 - mean_l_vars1) * (l_vars2 - mean_l_vars2))

    index1 = ((2 * mean_l_vars1 * mean_l_vars2) / (mean_l_vars1 ** 2 + mean_l_vars2 ** 2 + torch.finfo(torch.float32).eps))
    index2 = ((2 * std_l_vars1 * std_l_vars2) / (std_l_vars1 ** 2 + std_l_vars2 ** 2 + torch.finfo(torch.float32).eps))
    index3 = covar_l_vars / (std_l_vars1 * std_l_vars2 + torch.finfo(torch.float32).eps)

    return index1 * index2 * index3


def fbeta(*args, beta: float, num_classes: int, average: str = 'micro'):
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

    if average == 'micro':
        micro_tp = tp_set.sum()
        micro_fn = (row - tp_set).sum()
        micro_fp = (col - tp_set).sum()

        recall = micro_tp / (micro_tp + micro_fn + eps)
        precision = micro_tp / (micro_tp + micro_fp + eps)

        micro_f1 = (1 + beta ** 2) * recall * precision / ((beta ** 2) * precision + recall + eps)
        return micro_f1

    elif average == 'macro':
        macro_f1 = 0
        for tp, r, c in zip(tp_set, row, col):
            precision = tp / (c + eps)
            recall = tp / (r + eps)
            f1 = (1 + beta ** 2) * recall * precision / ((beta ** 2) * precision + recall + eps)
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

    assert (t1 := type(true)) == (t2 := type(predict)) == dict, \
        f'Expect type of given arguments are Dictionary, but receive {t1} and {t2}'

    def get_set(anno) -> set:
        """
        Calculate sample set in annotation with criteria.
        :param anno: Dictionary
        :return: set
        """
        ON, sample_set = False, set()

        assert len(samples := anno['sample']) == len(aux := anno['rhythm']), f'Lengths of keys must be same.'
        assert samples == sorted(samples), f'Key "sample" must be sorted.'

        for i, a in zip(samples, aux):
            if criteria in a:
                if not ON: ON = i
            else:
                if ON:
                    sample_set = sample_set.union(list(range(ON, i)))
                    ON = False
        if ON: sample_set = sample_set.union(list(range(ON, data_length)))
        return sample_set

    try:
        true_set, pred_set = get_set(true), get_set(predict)
    except KeyError:
        raise KeyError('Given arguments must have following keys, ["sample", "rhythm"].')
    except AssertionError as e:
        raise e
    else:
        inter = true_set.intersection(pred_set)
        pp, se = len(inter) / (len(pred_set) + EPS), len(inter)/(len(true_set) + EPS)
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

    assert len(true) % 2 == len(predict) % 2 == 0, f'Expect the length of each input argument to be even.'

    on, off = [], []
    for i, t in enumerate(true):
        set_to = off if i % 2 else on
        set_to += list(range(t-margin, t+margin))
    on, off = set(on), set(off)

    score = 0
    for i, p in enumerate(predict):
        find_set = off if i % 2 else on
        if p in find_set: score += 1
    score /= 2
    n_episode = max(len(true)//2, len(predict)//2)
    return score / n_episode
