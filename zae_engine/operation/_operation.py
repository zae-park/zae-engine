from typing import Union, List, Tuple, Optional
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.table import Table


class MorphologicalLayer(nn.Module):
    def __init__(self, ops: str, window_size: List[int]):
        super(MorphologicalLayer, self).__init__()
        try:
            assert len(ops) == len(window_size)
        except AssertionError:
            print("The lengths of the arguments must match.")

        class MorphLayer(nn.Module):
            def __init__(self, kernel_size, morph_type):
                super(MorphLayer, self).__init__()

                self.morph_type = morph_type

                self.conv = nn.Conv1d(1, kernel_size, kernel_size, bias=False, padding='same')
                kernel = torch.zeros((kernel_size, 1, kernel_size), dtype=torch.float)
                for i in range(kernel_size):
                    kernel[i][0][i] = 1
                self.conv.weight.data = kernel

            def forward(self, x):
                x = self.conv(x)
                if self.morph_type == 'erosion':
                    return torch.min(x, 1)[0].unsqueeze(1)
                elif self.morph_type == 'dilation':
                    return torch.max(x, 1)[0].unsqueeze(1)

        morph_list = []
        for op, ker in zip(ops, window_size):
            if op.lower() == 'c':
                morph_list += [MorphLayer(int(ker), 'dilation'), MorphLayer(int(ker), 'erosion')]
            elif op.lower() == 'o':
                morph_list += [MorphLayer(int(ker), 'erosion'), MorphLayer(int(ker), 'dilation')]
            else:
                print("Unexpected operation keyword.")
        self.post = nn.Sequential(*morph_list)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        temp = []
        for i in range(x.shape[1]):
            temp.append(self.post(x[:, i, :].unsqueeze(1)))

        return torch.concat(temp, dim=1)


def label_to_onoff(labels: Union[np.ndarray, torch.Tensor], sense: int = 2,
                   middle_only: bool = False, outside_idx: Optional = True) -> list:
    """
    Convert label sequence to onoff array.
    Receive the label(sequence of annotation for each point), return the on-off array.
    On-off array consists of [on, off, class] for exist beats. If there is no beat, return [].

    Input args:
        label: np.nd-array
                Sequence of annotation for each point
                Expected shape is [N, points] or [points] where N is number of data.
        sense: int
                The sensitivity value.
                Ignore beat if the (off - on) is less than sensitivity value.
        middle_only: bool
                Ignore both the left-most & right-most beats.
        outside_idx: int or float(nan)
                Outside index (default is np.nan).
                Fill on (or off) if beat is incomplete. only use for left-most or right-most.
                If middle_only is False, outside_idx is not used.

    Output args:
        Beat info matrix:
                Shape of matrix is (N, # of beats, 3) or (# of beats, 3) where N is number of data.
                Length of last dimension is 3 consists of [on, off, cls].
    """
    SINGLE = False
    if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
    if not len(labels.shape):
        raise IndexError('Receive empty array.')
    elif len(labels.shape) == 1:
        SINGLE = True
        labels = np.expand_dims(labels.copy(), 0)
    elif len(labels.shape) > 3:
        raise IndexError('Unexpected shape error.')
    else:
        assert len(labels.shape) == 2

    result = []
    for label in labels:
        cursor, res = 0, []
        n_groups = len(list(groupby(label)))
        groups = groupby(label)
        for i, (cls, g) in enumerate(groups):
            g_length = len(list(g))
            if cls:
                if i == 0:
                    if middle_only:
                        pass
                    else:
                        out_start = np.nan if outside_idx else 0
                        res.append([out_start, cursor + g_length - 1, int(cls)])
                elif i == n_groups - 1:
                    if middle_only:
                        pass
                    else:
                        out_end = np.nan if outside_idx else len(label) - 1
                        res.append([cursor, out_end, int(cls)])
                else:
                    if g_length < sense:
                        pass    # not enough length
                    else:
                        res.append([cursor, cursor + g_length - 1, int(cls)])
            else:
                pass    # class #0 is out of interest
            cursor += g_length
        if SINGLE: return res
        result.append(res)
    return result


def onoff_to_label(onoff: Union[np.ndarray, torch.Tensor], length: int = 2500) -> np.ndarray:
    """
    Return label sequence using onoff(arg #1).
    Receive the label(sequence of annotation for each point), return the on-off array.
    On-off array consists of [on, off, class] for exist beats. If there is no beat, return [].

    :param onoff: np.nd-array. Array of on-off. Expected shape is [N, [on, off, cls]] where N is number of beats.
    :param length: int. Length of label sequence. This value should be larger than maximum of onoff.
    :return: label
    """
    if isinstance(onoff, torch.Tensor): onoff = onoff.detach().numpy()
    label = np.zeros(length, dtype=int)
    if len(onoff.shape) == 1:
        return label
    elif len(onoff.shape) > 3:
        raise IndexError('Unexpected shape error.')
    else:
        assert len(onoff.shape) == 2
    if onoff.shape[-1] != 3:
        raise ValueError('Unexpected shape error.')

    for on, off, cls in onoff:
        on = 0 if np.isnan(on) else int(on)
        if np.isnan(off) or (int(off) >= length):
            label[on:] = cls
        else:
            label[on:int(off)+1] = cls

    return label


def sanity_check(onoff: Union[np.ndarray, torch.Tensor, list],
                 incomplete_only: bool = False,
                 tol_interval: int = 30,
                 tol_merge: int = 20) -> list:
    """
    Check how sane the onoff is.
    This function trims the inappropriate beats in onoff with the following steps,
        1. remove incomplete QRS (remove)
            If there exist the incomplete beats in onoff(args #1), remove them.
            This function assumes that at least one of the on-index and off-index of an incomplete beat is NaN.
        2. merge or remove close QRSs groups (merge)
            For complete beats, check how closely they are.
            If the gap btw 2 beats is less than tol_merge, both the bi-side beats are merged.
            If the gap btw 2 beats is more than tol_merge yet less than tol_interval,
             choose the widest beat and remove others.

    :param onoff: np.nd-array or torch.Tensor or list.
    :param incomplete_only: bool
        If True, return the onoff-array after step 1.
    :param tol_interval: int
        The tolerance value to find close beats.
    :param tol_merge: int
        The tolerance value to merge close beats.
    """

    def merge_and_remove(onoffs: np.ndarray, interval: np.ndarray, merge_th: int = 5, remove_th: int = 30):
        """
        Merge the beats closer than two thresholds.
        And then remove some beats except one of them whose larger than merge_th.
        This function operates as follows,
            1. Merge mode
                Accumulate the beats whose closer than merge_th.
                The new beat has width of cover of merged beats, and has class of the widest beat in the merged beats.
            2. Select mode
                After merge mode, compare remain beats and choose widest beat. Others are removed.
                If the new beat with larger interval than remove_th occurs during this mode,
                 accept it as a complete beat.
        :param onoffs: np.nd-array
        :param interval: np.nd-array
        :param merge_th: int
        :param remove_th: int
        :return: np.nd-array
        """
        # merge mode
        # interval이 5보다 작은 경우 QRS 폭을 점점 넓힙니다.
        # interval이 5보다 큰 경우 기존 QRS 폭으로 결정하고, 이후 다시 interval이 5 보다 작으면 새로운 QRS를 만들고 반복합니다.
        # QRS를 넓히는 과정 중에서 cls가 다른 onoff가 출현할 수 있습니다. 이 때 QRS 폭 결정 직전에 가장 넓은 폭을 갖는 cls를 선택합니다.
        merge_buff, vote = [onoffs[0]], [0] * 3
        vote[onoffs[0, -1] - 1] = onoffs[0, 1] - onoffs[0, 0]
        for i, gap in enumerate(interval):
            compare_onoff = onoffs[i + 1]
            compare_width = compare_onoff[1] - compare_onoff[0]

            if gap <= merge_th:
                vote[compare_onoff[-1] - 1] += compare_width  # 해당하는 cls에 맞게 폭 더하기
                merge_buff[-1][1] = compare_onoff[1]  # QRS 폭 넓히기
            else:
                merge_buff[-1][-1] = vote.index(max(vote)) + 1  # 기존 QRS의 cls 결정 및 폭 결정
                merge_buff.append(compare_onoff)
                vote = [0] * 3
                vote[compare_onoff[-1] - 1] = compare_width
        merge_buff[-1][-1] = vote.index(max(vote)) + 1
        remainder_interval = interval[interval > merge_th]

        # remove mode
        # merge 이후 남은 interval은 모두 5 이상, 30 이하
        # 첫 QRS를 기준으로 하고 QRS의 폭을 순차적으로 비교하며 탈락시킵니다.
        # 이 때, 탈락한 QRS로 인해 남은 QRS의 interval이 30 보다 커지면 또 하나의 온전한 QRS로 취급합니다.
        remove_buff = [merge_buff[0]]
        hidden_gap, last_gap = 0, 0
        for i, gap in enumerate(remainder_interval):
            compare_onoff = merge_buff[i + 1]  # 현재 QRS
            compare_width = compare_onoff[1] - compare_onoff[0]  # 현재 QRS의 폭
            last_width = remove_buff[-1][1] - remove_buff[-1][0]  # 직전 QRS의 폭

            if last_gap + hidden_gap + gap <= remove_th:  # 전체 gap = 직전 gap + 은닉 gap + 현재 gap
                if compare_width > last_width:
                    remove_buff[-1] = compare_onoff  # 현재 QRS를 선택
                    hidden_gap, last_gap = 0, 0  # 직전에 탈락으로 인한 gap 제거, 직전 gap 제거
                else:
                    # 이전 QRS 유지 및 현재 QRS 탈락
                    last_gap += hidden_gap + gap  # 직전 gap에 현재 gap 및 저장된 은닉 gap (탈락탈락이면 0이 아닐수도 있음) 누적.
                    hidden_gap = compare_width  # 은닉 gap에 현재 QRS 폭 update
            else:
                remove_buff.append(compare_onoff)
                hidden_gap, last_gap = 0, 0

        return np.concatenate(remove_buff, axis=0).reshape(-1, 3)

    if len(onoff) == 0:
        return []
    if not isinstance(onoff, np.ndarray):
        onoff = np.array(onoff)

    # 1 - remove incomplete QRS
    onoff = onoff[np.invert(np.isnan(onoff).any(1))].astype(int)
    if incomplete_only: return onoff.astype(int).tolist()

    # 2 & 3 - check QRS group
    intervals = onoff[1:, 0] - onoff[:-1, 1]  # i-th interval -> between i-th onoff i+1-th onoff
    closed_interval_map = ((intervals >= 0) & (intervals <= tol_interval))  # find intervals less than tol_interval.
    closed_interval_map = np.insert(closed_interval_map, (0, len(closed_interval_map)), (False, False))    # dummy
    suspect_idx = np.where(closed_interval_map == True)[0]
    suspect_idx = list(set(suspect_idx).union(set(suspect_idx-1)))
    close_interval_onoff = label_to_onoff(closed_interval_map, sense=1)  # [on of closed set, off, 1]

    new_onoff = []
    for closed_on, closed_off, _ in close_interval_onoff:
        onoff_set = onoff[closed_on-1:closed_off+1]         # (n+1) onoffs in closed set
        interval_set = intervals[closed_on-1:closed_off]     # n intervals in closed set
        final_onoff = merge_and_remove(onoffs=onoff_set, interval=interval_set,
                                       merge_th=tol_merge, remove_th=tol_interval)
        new_onoff.append(final_onoff)

    for i, oo in enumerate(onoff):
        if i not in suspect_idx:
            new_onoff.append(oo.reshape(-1, 3))

    if len(new_onoff) != 0:
        new_onoff = sorted(np.concatenate(new_onoff), key=lambda x: x[0].tolist())

    return new_onoff


def find_nearest(arr: Union[np.ndarray, torch.Tensor], value: int):
    """
    Find the nearest value and its index.
    :param arr: 1d-array.
    :param value: reference value.
    :return: index of nearest, value of nearest
    """
    if isinstance(arr, torch.Tensor): arr = arr.numpy()

    i_gap = np.searchsorted(arr, value)

    if i_gap == 0:
        return i_gap, arr[0]                # arr의 최소값보다 작은 value
    elif i_gap == len(arr):
        return len(arr) - 1, arr[-1]               # arr의 최대값보다 큰 value
    else:
        left, right = arr[i_gap - 1], arr[i_gap]
        if abs(value - left) <= abs(right - value):
            return i_gap - 1, left
        else:
            return i_gap, right


def draw_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor],
                          y_hat: Union[np.ndarray, torch.Tensor],
                          num_classes: int):
    """
    Compute confusion matrix.
    Both the y_true and y_hat have data type as integer, and match in shape.

    :param y_true: Union[np.nd-array, torch.Tensor]
    :param y_hat: Union[np.nd-array, torch.Tensor]
    :param num_classes: int
    :return: confusion matrix with 2-D nd-array.
    """

    assert len(y_true) == len(y_hat), f'length unmatched: arg #1 {len(y_true)} =/= arg #2 {len(y_hat)}'
    canvas = np.zeros((num_classes, num_classes))

    for true, hat in zip(y_true, y_hat):
        canvas[true, hat] += 1

    return canvas


def print_confusion_matrix(confusion_matrix: np.ndarray,
                           cell_width: Optional[int] = 4,
                           class_name: Union[List, Tuple] = None,
                           frame: Optional[bool] = True):
    """
    Printing given confusion matrix.
    Printing width is customizable with cell_width, but height is not.
    The names of rows and columns are customizable with class_name.
    Note that the length of class_name must be matched with the length of the confusion matrix.
    :param confusion_matrix: np.nd-array
    :param cell_width: int, optional
    :param class_name: Union[List[str], Tuple[str]], optional
    :param frame: bool, optional
    :return:
    """
    box_frame = box.SIMPLE if frame else None

    table = Table(show_header=True,
                  header_style="bold magenta",
                  box=box_frame,
                  leading=1,
                  show_edge=True,
                  show_lines=True,
                  min_width=64
                  )

    console = Console()
    table.title = '\n confusion_matrix'

    if class_name is not None:
        assert len(class_name) == confusion_matrix.shape[-1],\
            f'Unmatched classes number class_name {len(class_name)} =/= number of class {confusion_matrix.shape[-1]}'

        class_name = [''] + class_name
        for i, name in enumerate(class_name):
            if i == 0:
                table.add_column('', justify='center', style='green', min_width=cell_width, max_width=cell_width)
            else:
                table.add_column(name, justify='center', min_width=cell_width, max_width=cell_width)

        for i, row in enumerate(confusion_matrix):
            row_with_index = [class_name[i+1]] + list(map(lambda x: str(int(x)), row.tolist()))
            row_with_index[i + 1] = f'[bold cyan]{row_with_index[i + 1]}[bold cyan]'
            table.add_row(*row_with_index)

    else:
        for col in range(confusion_matrix.shape[-1] + 1):
            if col == 0:
                table.add_column('', justify='center', style='green', min_width=cell_width, max_width=cell_width)
            else:
                table.add_column('P' + str(col - 1), justify='center', min_width=cell_width, max_width=cell_width)

        for i, row in enumerate(confusion_matrix):
            row_with_index = [f'T{i}'] + list(map(lambda x: str(int(x)), row.tolist()))
            row_with_index[i + 1] = f'[bold cyan]{row_with_index[i + 1]}[bold cyan]'
            table.add_row(*row_with_index)

    table.caption = 'row : [green]Actual[/green] column : [purple]Prediction[/purple]'

    console.print(table)
