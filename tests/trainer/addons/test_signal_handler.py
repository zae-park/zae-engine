import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import torch
from torch import nn, optim
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import SignalHandlerAddon


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class DummyTrainer(Trainer):
    def train_step(self, batch):
        return {"loss": torch.tensor(0.0)}

    def test_step(self, batch):
        return {"loss": torch.tensor(0.0)}


class TestSignalHandlerAddon(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_save_path = os.path.join(self.temp_dir, "killed_model.ckpt")
        self.log_file_path = os.path.join(self.temp_dir, "signal_handler.log")

        self.model = DummyModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # Trainer with SignalHandlerAddon
        MyTrainerWithSignal = DummyTrainer.add_on(SignalHandlerAddon)
        self.trainer = MyTrainerWithSignal(
            model=self.model,
            device=torch.device("cpu"),
            mode="train",
            optimizer=self.optimizer,
            scheduler=None,
            signal_save_path=self.model_save_path,
            signal_log_path=self.log_file_path,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("signal.signal")
    def test_signal_registration(self, mock_signal):
        """
        Test that signal handlers (SIGINT, SIGTERM) are registered properly.
        """
        MyTrainerWithSignal = DummyTrainer.add_on(SignalHandlerAddon)
        trainer = MyTrainerWithSignal(
            model=self.model,
            device=torch.device("cpu"),
            mode="train",
            optimizer=self.optimizer,
            scheduler=None,
        )
        # mock_signal가 SIGINT와 SIGTERM 등록을 받았는지 확인
        calls = [call[0][0] for call in mock_signal.call_args_list]
        self.assertIn(2, calls)  # 2 == signal.SIGINT
        self.assertIn(15, calls)  # 15 == signal.SIGTERM

    @patch("sys.exit")
    def test_signal_handler_action(self, mock_sys_exit):
        """
        Test the _signal_handler internal logic:
        - Logs the signal
        - Saves the model
        - Writes to log file
        - Exits the process
        """
        # 모델 저장 함수를 모의
        with patch.object(self.trainer, "save_model", wraps=self.trainer.save_model) as mock_save_model:
            # _signal_handler 메서드를 직접 호출
            signum = 2  # signal.SIGINT
            self.trainer._signal_handler(signum, None)

            # 모델이 저장되었는지
            mock_save_model.assert_called_with(self.model_save_path)
            self.assertTrue(os.path.exists(self.model_save_path))

            # 로그 파일 내용 확인
            self.assertTrue(os.path.exists(self.log_file_path))
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Caught signal: SIGINT", content)
                self.assertIn("Saving model to:", content)
                self.assertIn("Model saved successfully.", content)
                self.assertIn("Terminating training process...", content)

            # 프로세스 종료
            mock_sys_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
