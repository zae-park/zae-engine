import os
from typing import Union, Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def seg_plot(waveform: Union[np.ndarray, List],
             true: Optional[Union[np.ndarray, List, Tuple]] = None,
             pred: Optional[Union[np.ndarray, List, Tuple]] = None,
             rpeak: Optional[Union[np.ndarray, List, Tuple]] = None,
             class_names: Union[List, Tuple] = None,
             title: Optional[str] = 'Unknown',
             minmax: Optional[bool] = False,
             save_path: Optional[str] = None,
             figure_number: Optional[int] = 1030):

    def type_check(val):
        if val is not None:
            val = np.array(val) if not isinstance(val, np.ndarray) else val.copy()
            assert len(val.shape) == 1, f'Shape error. Expect 1-D array but receive {len(val.shape)}-D array.'
            return val

    wave, true, pred, rpeak = type_check(waveform), type_check(true), type_check(pred), type_check(rpeak)

    if minmax:
        M, m = np.max(wave), np.min(wave)
        wave = ((wave - m) / (M - m))

    plot_dict = {'true': {'color': 'r', 'seg': true}, 'pred': {'color': 'b', 'seg': pred}}

    cls_set = set()
    for t in [true, pred]:
        try:
            cls_set = cls_set.union(set(t))
        except TypeError:
            pass
    cls_set = tuple(cls_set)

    legend = ['wave']
    plt.figure(figure_number, figsize=(10, 4))
    plt.plot(wave * 2, 'k', linewidth=1)

    # set appropriate y-lim.
    ylim = [-1.0, 2.0] if minmax else [-9.0, 6.0]
    plt.ylim([ylim[0] - 0.2, ylim[1] + 0.2])      # 0.2 margin

    for k, v in plot_dict.items():
        if v['seg'] is None: continue
        plt.plot(ylim[0]+abs(ylim[1]+ylim[0])*v['seg']/np.max(cls_set), v['color'], linewidth=0.5)
        legend.append(k)

    if rpeak is not None and len(rpeak) > 0:
        for rp in rpeak:
            plt.axvline(rp, linestyle=(0, (5, 3)), color='forestgreen', linewidth=0.2)

    plt.legend(legend)

    if class_names is None:
        class_names = tuple([None] * (np.max(cls_set)+1)) if cls_set else ()
    n_class = len(class_names)

    for i_cls in range(n_class):
        h = (ylim[0]) + abs(ylim[1]+ylim[0])*i_cls/(n_class-1)
        plt.axhline(h, linestyle=(0, (5, 2)), color='grey', linewidth=0.5)
        plt.text(0, h, class_names[i_cls], fontsize=7, ha='right')
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, title + '.png'), dpi=80)
    else:
        plt.show()
    plt.close(figure_number)
