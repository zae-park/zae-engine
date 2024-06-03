import os
import time
from PIL import Image
import urllib.request
from typing import Union, Tuple

import wfdb
import numpy as np
import nibabel as nib
from nibabel.testing import data_path


IMAGE_FORMAT = ["png", "jpg", "jpeg", "til"]


def image_from_url(url: str, save_dst: str = None) -> Union[None, Image.Image]:

    save_mode = False if save_dst is None else True

    if save_dst is None:
        save_dst = os.path.join(os.getcwd(), str(time.time()).replace(".", "") + ".png")
    else:
        if ext := os.path.splitext(save_dst)[-1] not in IMAGE_FORMAT:
            raise AssertionError(f'Invalid extension. Expect one of {IMAGE_FORMAT}, but receive "{ext}".')
    urllib.request.urlretrieve(url, save_dst)
    if save_mode:
        return
    else:
        img = Image.open(save_dst)
        os.remove(save_dst)
        return img


def example_ecg(beat_idx: int = None) -> Tuple[np.ndarray, ...]:
    """
    Load 10 second ecg recording \& annotation from example data with sampling frequency 250Hz.
    The example data from LUDB(https://www.physionet.org/content/ludb/1.0.1/).
    The '*.zea' file includes QRS complex information.
    If argument 'beat_idx' is None(default), return the 10 second recording and label sequence.
    If not, return the recording, r-peak index, and beat type for beat_idx'th beat.
    :param beat_idx: int, optional
    The index of beat in data.
    Note that the value cannot be more than maximum index of beats in data (12).
    If this parameter is not specified, run for 10 second data.
    :return:
    If beat_idx was given, return recording, r-peak index, and beat type for beat_idx'th beat.
    If not, return recording, label sequence for 10 sec data.
    """
    lookup = {"N": 1, "A": 2}

    ex_path = os.path.join(os.path.dirname(__file__), "__sample/sample_data")
    recording = wfdb.rdsamp(ex_path, return_res=32)[0].squeeze()  # nd-array, [2500, ]
    anno = wfdb.rdann(ex_path, "zae")
    samp, sym = anno.sample.tolist(), anno.symbol
    if sym[0] != "*":
        samp.insert(0, 0)
        sym.insert(0, "*")
    if sym[-1] != "*":
        samp.append(2499)
        sym.append("*")

    assert len(sym) == len(samp), "The length of samples and symbols is not matched. The annotation file is insane."
    assert len(sym) % 3 == 0, "Invalid symbol. Please check out the first & last symbol of annotation."

    n_qrs = len(sym) // 3  # The number of QRS complexes.
    if beat_idx is None:
        label = np.zeros_like(recording, dtype=np.int32)
        for i_qrs in np.arange(len(sym))[1::3]:
            label[samp[i_qrs - 1] : samp[i_qrs + 1]] = lookup[s] if (s := sym[i_qrs]) in lookup.keys() else 3
        return recording, label
    else:
        assert beat_idx < n_qrs, f"The maximum value os beat_idx is {n_qrs}. But {beat_idx} was provided."
        qrs_chunk = recording[samp[3 * beat_idx] : samp[3 * beat_idx + 2]]
        sym = sym[3 * beat_idx + 1]
        qrs_loc = samp[3 * beat_idx + 1] - samp[3 * beat_idx]
        return qrs_chunk, qrs_loc, sym


def example_mri() -> np.ndarray:
    example_path = os.path.join(data_path, "example4d.nii.gz")
    proxy = nib.load(example_path)
    if "get_fdata" in proxy.__dir__():
        return proxy.dataobj


if __name__ == "__main__":
    res = example_mri()
