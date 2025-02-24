import os
import signal
import sys
from datetime import datetime
from typing import Type, Optional

from .core import AddOnBase
from .._trainer import T
from ..addons import ADDON_CASE_TRIGGER


class SignalHandlerAddon(AddOnBase):
    addon_case = ADDON_CASE_TRIGGER  # Role of event trigger
    """
    Add-on to handle external termination signals (e.g., SIGTERM, SIGINT) during training.

    This add-on automatically logs the reason for process termination, prints console messages,
    and optionally saves the model state when a termination signal is caught. If the user does
    not specify a log path, a default file named 'log_SIGHANDLER_{YYYYMMDD_HHMMSS}.log' will
    be created. The user can also specify a path to save the model upon receiving the signals.

    Parameters
    ----------
    signal_save_path : str, optional
        The path to save the model when a termination signal is caught.
        If None, the model is not saved upon termination. Default is None.
    signal_log_path : str, optional
        The path to save the log of caught signals and additional info.
        If None, a file named 'log_SIGHANDLER_{YYYYMMDD_HHMMSS}.log' will be automatically created.
        Default is None.

    Methods
    -------
    _signal_handler(signum, frame)
        Internal method that handles the caught signal. Logs the event, saves the model if specified,
        and terminates the process.

    Notes
    -----
    - This add-on registers handlers for SIGINT (e.g., Ctrl + C) and SIGTERM (kill -15).
    - Other signals (e.g., SIGABRT, SIGKILL) are not handled here by default, but you could add them
      if needed.
    - MemoryError(Out-of-memory) is an exception rather than a signal, so it requires separate handling
      via `try-except` or a custom exception hook.

    Examples
    --------
    Adding SignalHandlerAddon to a Trainer:

    >>> from zae_engine.trainer import Trainer
    >>> from zae_engine.trainer.addons import SignalHandlerAddon

    >>> MyTrainer = Trainer.add_on(SignalHandlerAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     signal_save_path='./checkpoints/model_after_kill.ckpt',
    >>>     signal_log_path='./logs/signal_handler.log'
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader, valid_loader=valid_loader)
    """

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        class TrainerWithSignalHandler(base_cls):
            def __init__(
                self, *args, signal_save_path: Optional[str] = None, signal_log_path: Optional[str] = None, **kwargs
            ):
                super().__init__(*args, **kwargs)

                # 1) 모델 저장 경로
                self.signal_save_path = signal_save_path

                # 2) 로그 경로가 지정되지 않았다면, 기본 파일명 생성
                if signal_log_path is None:
                    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    signal_log_path = f"log_SIGHANDLER_{dt_str}.log"
                    print(f"[SignalHandlerAddon] No log path provided. " f"Using default log file: {signal_log_path}")
                self.signal_log_path = signal_log_path

                # 관리할 시그널 (KeyboardInterrupt: SIGINT, 프로세스 강제 종료: SIGTERM)
                handled_signals = [signal.SIGINT, signal.SIGTERM]
                for sig in handled_signals:
                    signal.signal(sig, self._signal_handler)

            def _signal_handler(self, signum, frame):
                """
                Internal method to handle termination signals.
                Logs the reason and saves the model if a path is provided.
                Finally, terminates the training process.

                Parameters
                ----------
                signum : int
                    The signal number (e.g., SIGINT, SIGTERM).
                frame : FrameType
                    The current stack frame (ignored here).
                """
                signum_name = signal.Signals(signum).name
                message = f"\n[SignalHandlerAddon] Caught signal: {signum_name}"

                def log_to_file(text: str):
                    """Helper function to write text to the configured log file."""
                    if self.signal_log_path:
                        log_dir = os.path.dirname(self.signal_log_path)
                        if log_dir:
                            os.makedirs(log_dir, exist_ok=True)
                        with open(self.signal_log_path, "a", encoding="utf-8") as f:
                            f.write(text + "\n")

                # 1) 콘솔 출력 + 로그 기록
                print(message)
                log_to_file(message)

                # 2) 모델 저장
                if self.signal_save_path:
                    save_msg = f"[SignalHandlerAddon] Saving model to: {self.signal_save_path}"
                    print(save_msg)
                    log_to_file(save_msg)

                    try:
                        self.save_model(self.signal_save_path)
                        done_msg = "[SignalHandlerAddon] Model saved successfully."
                        print(done_msg)
                        log_to_file(done_msg)
                    except Exception as e:
                        err_msg = f"[SignalHandlerAddon] Error saving model: {e}"
                        print(err_msg)
                        log_to_file(err_msg)

                # 3) 로그 파일 경로 안내
                path_msg = f"[SignalHandlerAddon] Signal log is stored at: {self.signal_log_path}"
                print(path_msg)
                log_to_file(path_msg)

                # 4) 프로세스 종료 메시지
                exit_msg = "[SignalHandlerAddon] Terminating training process..."
                print(exit_msg)
                log_to_file(exit_msg)

                # 5) 프로세스 종료
                sys.exit(1)

        return TrainerWithSignalHandler
