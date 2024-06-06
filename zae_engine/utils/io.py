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
    Load a 10-second ECG recording and annotation from LUDB dataset [1], which has a sampling frequency of 250Hz.

    The '*.zae' file in package contains QRS complex information from raw data[1] and beat locations via algorithm.
    If the argument 'beat_idx' is None (default), the function returns the 10-second recording and label sequence.
    If 'beat_idx' is specified, the function returns the segment of the recording, the location, and the symbol of beat.

    Parameters
    ----------
    beat_idx : int, optional
        The index of the beat in the data.
        The value cannot be greater than the maximum index of beats in the data (12).
        If this parameter is not specified, the function returns the 10-second data.

    Returns
    -------
    signal : np.ndarray
        The 10-second ECG recording.
        If 'beat_idx' is specified, this is the segment of the recording around the specified beat.
    label : np.ndarray or int
        If 'beat_idx' is None, this is the label sequence for the 10-second data.
        If 'beat_idx' is specified, this is the type of the specified beat.
    loc : None or int
        The location of the specified beat if 'beat_idx' is specified.
        Otherwise, this is None.

    References
    ----------
    .. [1]  Kalyakulina, A., Yusipov, I., Moskalenko, V., et al.
            Lobachevsky University Electrocardiography Database (version 1.0.0). PhysioNet. (2020).
            https://doi.org/10.13026/qweb-sr17.

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
            label[samp[i_qrs - 1]: samp[i_qrs + 1]] = lookup[s] if (s := sym[i_qrs]) in lookup.keys() else 3
        return recording, label
    else:
        assert beat_idx < n_qrs, f"The maximum value os beat_idx is {n_qrs}. But {beat_idx} was provided."
        qrs_chunk = recording[samp[3 * beat_idx]: samp[3 * beat_idx + 2]]
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
