from typing import Union, Tuple, List
import numpy as np
import torch

from .iou import giou
from .confusion import print_confusion_matrix
from .count import f_beta_from_mat as f_beta_

from zae_engine.operation import arg_nearest, RunLengthCodec, Run, RunList


class BijectiveMetrix:
    """
    Compute bijective confusion matrix of given sequences.

    The surjective operation projects every run in prediction onto label and checks IoU.
    The injective projection is the opposite.

    Parameters
    ----------
    prediction : Union[np.ndarray, torch.Tensor]
        Sequence of predicted labels.
    label : Union[np.ndarray, torch.Tensor]
        Sequence of true labels.
    num_classes : int
        The number of classes.
    th_run_length : int, optional
        The minimum length of runs to be considered. Default is 2.

    Attributes
    ----------
    bijective_run_pair : torch.Tensor
        The bijective pairs of runs between prediction and label.
    injective_run_pair : torch.Tensor
        The injective mapping from prediction to label.
    surjective_run_pair : torch.Tensor
        The surjective mapping from label to prediction.
    injective_mat : np.ndarray
        The confusion matrix from injective mapping.
    surjective_mat : np.ndarray
        The confusion matrix from surjective mapping.
    bijective_mat : np.ndarray
        The confusion matrix from bijective mapping.
    bijective_f1 : float
        The F1 score from bijective mapping.
    injective_f1 : float
        The F1 score from injective mapping.
    surjective_f1 : float
        The F1 score from surjective mapping.
    bijective_acc : float
        The accuracy from bijective mapping.
    injective_acc : float
        The accuracy from injective mapping.
    surjective_acc : float
        The accuracy from surjective mapping.
    """

    def __init__(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        label: Union[np.ndarray, torch.Tensor],
        num_classes: int,
        th_run_length: int = 2,
    ):
        self.num_classes = num_classes
        self.eps = torch.finfo(torch.float32).eps
        assert prediction.shape == label.shape, f"Unmatched shape error. {prediction.shape} =/= {label.shape}"
        assert len(label.shape) == 1, f"Unexpected shape error. Expect 1-D array but receive {label.shape}."
        assert prediction.size > 0, "Prediction array is empty."
        assert label.size > 0, "Label array is empty."

        # Initialize RunLengthCodec
        codec = RunLengthCodec(base_class=0, merge_closed=True, tol_merge=5)
        # Encode prediction and label
        self.pred_run_list = codec.encode(prediction.tolist(), sense=th_run_length)
        self.pred_run_list.all_runs = self.pred_run_list.filtered()
        self.pred_runs = self.pred_run_list.all_runs
        self.pred_array = np.array(codec.decode(self.pred_run_list))
        self.label_run_list = codec.encode(label.tolist(), sense=th_run_length)
        self.label_run_list.all_runs = self.label_run_list.filtered()
        self.label_runs = self.label_run_list.all_runs
        self.label_array = np.array(codec.decode(self.label_run_list))

        self.bijective_run_pair, self.injective_run_pair, self.surjective_run_pair = self.run_pairing()
        # Injective: pred --projection-> label
        self.injective_mat = self.map_and_confusion(self.pred_runs, self.label_array).transpose()
        self.injective_count = self.injective_mat.sum()
        # Surjective: pred <-projection-- label
        self.surjective_mat = self.map_and_confusion(self.label_runs, self.pred_array)
        self.surjective_count = self.surjective_mat.sum()
        # Bijective: using run_pair
        self.bijective_mat = self.bijective_confusion()
        self.bijective_count = self.bijective_mat.sum()
        self.bijective_f1 = f_beta_(self.bijective_mat[1:, 1:], beta=1, num_classes=self.num_classes)
        self.injective_f1 = f_beta_(self.injective_mat[1:, 1:], beta=1, num_classes=self.num_classes)
        self.surjective_f1 = f_beta_(self.surjective_mat[1:, 1:], beta=1, num_classes=self.num_classes)
        self.bijective_acc = self.bijective_mat[1:, 1:].trace() / (self.bijective_mat[1:, 1:].sum() + self.eps)
        self.injective_acc = self.injective_mat[1:, 1:].trace() / (self.injective_mat[1:, 1:].sum() + self.eps)
        self.surjective_acc = self.surjective_mat[1:, 1:].trace() / (self.surjective_mat[1:, 1:].sum() + self.eps)

    def run_pairing(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pair runs between prediction and label sequences.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing bijective, injective, and surjective run pairs.
        """
        injective_run_pair = []
        surjective_run_pair = []

        if self.label_runs and self.pred_runs:
            pred_run_starts = torch.tensor([run.start_index for run in self.pred_runs])
            label_run_starts = torch.tensor([run.start_index for run in self.label_runs])
            for i_p, p_run in enumerate(self.pred_runs):
                p_run_values = np.array([p_run.start_index, p_run.end_index, p_run.value])
                i_nearest, _ = arg_nearest(label_run_starts, p_run.start_index)  # find nearest label run
                l_run = self.label_runs[i_nearest]
                l_run_values = np.array([l_run.start_index, l_run.end_index, l_run.value])
                iou = giou(p_run_values[:-1], l_run_values[:-1])[0]
                if iou > 0.5:
                    injective_run_pair.append([i_p, i_nearest, p_run_values[-1], l_run_values[-1]])

            for i_l, l_run in enumerate(self.label_runs):
                l_run_values = np.array([l_run.start_index, l_run.end_index, l_run.value])
                i_nearest, _ = arg_nearest(pred_run_starts, l_run.start_index)  # find nearest pred run
                p_run = self.pred_runs[i_nearest]
                p_run_values = np.array([p_run.start_index, p_run.end_index, p_run.value])
                iou = giou(l_run_values[:-1], p_run_values[:-1])[0]
                if iou > 0.5:
                    surjective_run_pair.append([i_nearest, i_l, p_run_values[-1], l_run_values[-1]])

        bijective_run_pair = [inj for inj in injective_run_pair if inj in surjective_run_pair]
        injective_run_pair = (
            torch.tensor(injective_run_pair, dtype=torch.int)
            if injective_run_pair
            else torch.empty((0, 4), dtype=torch.int)
        )
        surjective_run_pair = (
            torch.tensor(surjective_run_pair, dtype=torch.int)
            if surjective_run_pair
            else torch.empty((0, 4), dtype=torch.int)
        )
        bijective_run_pair = (
            torch.tensor(bijective_run_pair, dtype=torch.int)
            if bijective_run_pair
            else torch.empty((0, 4), dtype=torch.int)
        )
        return bijective_run_pair, injective_run_pair, surjective_run_pair

    def map_and_confusion(self, x_runs: List[Run], y_array: np.ndarray) -> np.ndarray:
        """
        Map runs to label array and compute confusion matrix.

        Parameters
        ----------
        x_runs : List[Run]
            List of runs to map.
        y_array : np.ndarray
            Label array to map onto.

        Returns
        -------
        np.ndarray
            The confusion matrix.
        """
        confusion_mat = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        for run in x_runs:
            start, end, x = run.start_index, run.end_index, run.value
            y_values = y_array[start : end + 1]
            if len(y_values) > 0:
                y = int(np.bincount(y_values).argmax())
            else:
                y = 0  # base class
            confusion_mat[x, y] += 1
        return confusion_mat

    def bijective_confusion(self) -> np.ndarray:
        """
        Compute confusion matrix using bijective mapping.

        Returns
        -------
        np.ndarray
            The confusion matrix.
        """
        confusion_mat = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        for _, _, p, l in self.bijective_run_pair:
            confusion_mat[l, p] += 1
        return confusion_mat

    def summary(self, class_name: Union[Tuple, List] = None):
        """
        Print a summary of the bijective metrics.

        Parameters
        ----------
        class_name : Union[Tuple, List], optional
            Names of the classes for display. Default is None.
        """
        print(
            "\t\t# of samples in bijective confusion mat -> Bi: {bi} / Inj: {inj} / Sur: {sur}".format(
                bi=self.bijective_count, inj=self.injective_count, sur=self.surjective_count
            )
        )
        print(
            "\t\tF-beta score from bijective metric -> Bi: {bi:.2f}% / Inj: {inj:.2f}% / Sur: {sur:.2f}%".format(
                bi=100 * self.bijective_f1, inj=100 * self.injective_f1, sur=100 * self.surjective_f1
            )
        )
        print(
            "\t\tAccuracy from bijective metric -> Bi: {bi:.2f}% / Inj: {inj:.2f}% / Sur: {sur:.2f}%".format(
                bi=100 * self.bijective_acc, inj=100 * self.injective_acc, sur=100 * self.surjective_acc
            )
        )
        print_confusion_matrix(self.bijective_mat, class_name=class_name)
        num_catch = self.bijective_mat[1:, 1:].sum()
        print(
            "Beat Acc : %d / %d -> %.2f%%"
            % (num_catch, self.bijective_count, 100 * num_catch / (self.bijective_count + self.eps))
        )
        print()
