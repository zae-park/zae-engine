import pickle

from abc import abstractmethod
from typing import Optional, Iterable

import numpy as np
import torch


class CallbackInterface:
    """
    Handler for cache and execute callback function or functions
    :params callbacks: Union[Callable, Iterable], Optional
    """

    def __init__(self, callback_step: int = None, callback_epoch: int = None):
        if callback_step is None and callback_epoch is None:
            raise ValueError('both callback_step and callback_epoch is None')
        else:
            self.callback_epoch = callback_epoch
            self.callback_step = callback_step
        self.is_step = None
        self.is_epoch = None

    # TODO: event checker
    def is_triggered(self) -> bool:
        if self.check_epoch_event():
            self.is_epoch = True
            self.is_step = False
            return True
        elif self.check_step_event():
            self.is_step = True
            self.is_epoch = False
            return True
        else:
            return False

    # TODO: check -> should be decorator?
    def check_step_event(self) -> bool:
        if self.callback_step is not None:
            return self.attr.progress_checker.get_step() % self.callback_step == 0
        else:
            return False

    # TODO: check -> should be decorator?
    def check_epoch_event(self) -> bool:
        if self.callback_epoch is not None:
            return self.attr.progress_checker.get_epoch() % self.callback_epoch == 0
        else:
            return False

    def __call__(self, attr):
        self.attr = attr
        if self.is_triggered():
            self.callback(attr)

    @abstractmethod
    def callback(self, attr):
        pass


class EpochStepChecker(CallbackInterface):

    def __init__(self, callback_step: int, callback_epoch: int):
        super(EpochStepChecker, self).__init__(callback_step=callback_step, callback_epoch=callback_epoch)

    def callback(self, attr):
        print(f'progress_checker_steps : {self.attr.progress_checker.get_step()}')
        print(f'progress_checker_epochs : {self.attr.progress_checker.get_epoch()}')


class NeptuneCallback(CallbackInterface):
    """
    Logger for upload training results at neptune.io

    :params keys Iterable
    :params neptune_run Run
    """

    def __init__(
            self,
            *args,
            callback_step: int,
            callback_epoch: Optional[int] = None
    ):
        super(NeptuneCallback, self).__init__(callback_step=callback_step, callback_epoch=callback_epoch)
        self.log_keys = args

    def reduce(self, value):
        return np.mean(value) if self.is_epoch else value[-1]

    def callback(self, attr):
        for k in self.log_keys:
            if attr.mode == 'train':
                attr.web_logger.log(f'train/{k}', self.reduce(attr.log_train[k]))
            else:
                attr.web_logger.log(f'valid/{k}', self.reduce(attr.log_test[k]))


# ------------------------------------- Legacy start ------------------------------------- #


class CKPTSaver(CallbackInterface):
    """
    Callback class for saving training checkpoint

    :params file_name: file_name & path where the result file saved
    """

    def __init__(
            self,
            file_name: str = 'checkpoint.pth',
            callback_step: Optional[int] = None,
            callback_epoch: Optional[int] = None
    ):
        super(CKPTSaver, self).__init__(callback_step=callback_step, callback_epoch=callback_epoch)
        self.file_name = file_name

    def callback(self, attr):
        ckpt = {
            'state_dict': attr.model.state_dict(),
            'optimizer': attr.optimizer.state_dict() if attr.optimizer is not None else None,
            'scheduler': attr.scheduler.state_dict() if attr.scheduler is not None else None,
            'log_train': attr.log_train
        }
        if attr.valid_loader is not None:
            attr.toggle()
            attr.run_epoch(attr.valid_loader)
            for k in attr.log_test.keys():
                ckpt['log_test'] = attr.log_test[k]
            attr.log_test.clear()
            attr.toggle()
            attr.n_valid_data = len(attr.valid_loader.dataset)

        torch.save(ckpt, self.file_name)


class ResultSaver(CallbackInterface):
    """
    Call back class for saving result as a file

    :params keys: keys defined train, test step method
    :params file_name: file_name & path where the result file saved
    """

    def __init__(
            self,
            keys: Iterable,
            file_name: str = 'result.pkl',
            callback_step: Optional[int] = None,
            callback_epoch: Optional[int] = None
    ):
        super(ResultSaver, self).__init__(callback_step=callback_step, callback_epoch=callback_epoch)
        self.file_name = file_name
        self.keys = keys

    def callback(self, attr):
        result = {}
        for k in self.keys:
            result['log_train'] = attr.log_train[k]
            if attr.valid_loader is not None:
                attr.toggle()
                attr.run_epoch(attr.valid_loader)
                for k in self.keys:
                    result['log_test'] = attr.log_test[k]
                attr.log_test.clear()
                attr.toggle()
                attr.n_valid_data = len(attr.valid_loader.dataset)

        if result:
            with open(self.file_name, 'wb') as pkl:
                pickle.dump(result, pkl)



# ------------------------------------- Legacy end ------------------------------------- #
