import os
from typing import Tuple

import wfdb
import numpy as np


def load_example(beat_idx: int = None) -> Tuple[np.ndarray, ...]:
    """
    Load example 10 second data from LUDB.
    If argument 'beat_idx' is None(default), return the 10 second recording and label sequence.
    If not, return the recording, r-peak index, and beat type for beat_idx'th beat.
    :param beat_idx: str, optional
        The index of beat in data.
        Note that the value cannot be more than maximum index of beats in data (12).
        If this parameter is not specified, run for 10 second data.
    :return:
        If beat_idx was given, return recording, r-peak index, and beat type for beat_idx'th beat.
        If not, return recording, label sequence for 10 sec data.
    """
    resource_path = os.path.join(os.path.dirname(__file__), 'resource/sample_data')
    recording = wfdb.rdsamp(resource_path, return_res=32)[0].squeeze()     # nd-array, [2500, ]
    anno = wfdb.rdann(resource_path, 'common')
    samples, symbols = anno.sample.tolist(), anno.symbol
    if symbols[0] != '*':
        samples.insert(0, 0)
        symbols.insert(0, '*')
    if symbols[-1] != '*':
        samples.append(2499)
        symbols.append('*')

    assert len(symbols) == len(samples)
    assert len(symbols) % 3 == 0
    n_peak = len(symbols) // 3

    if beat_idx is None:
        label = np.zeros_like(recording, dtype=np.int32)
        for i_peak in range(n_peak):
            if symbols[3*i_peak+1] == 'N':
                label[samples[3*i_peak]:samples[3*i_peak+2]] = 1
            elif symbols[3*i_peak+1] == 'A':
                label[samples[3*i_peak]:samples[3*i_peak+2]] = 2
            else:
                label[samples[3*i_peak]:samples[3*i_peak+2]] = 3
        return recording, label
    else:
        assert beat_idx < n_peak
        chunk = recording[samples[3*beat_idx]:samples[3*beat_idx+2]]
        sym = symbols[3*beat_idx+1]
        peak_loc = samples[3*beat_idx+1] - samples[3*beat_idx]
        return chunk, peak_loc, sym
