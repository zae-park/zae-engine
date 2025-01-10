import unittest
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import PrecisionMixerAddon


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class CustomTrainer(Trainer):
    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        x, y = batch
        outputs = self.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        return {"loss": loss}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        x, y = batch
        outputs = self.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        return {"loss": loss}


class TestPrecisionMixerAddon(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        # Generate some random data
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        self.loader = DataLoader(dataset, batch_size=10, shuffle=True)

    def test_cpu_device_error(self):
        """Test that using a CPU-only device raises an error."""
        if torch.cuda.is_available():
            self.skipTest("Skipping CPU-only test on a CUDA-enabled machine.")

        with self.assertRaises(ValueError) as context:
            trainer_cls = CustomTrainer.add_on(PrecisionMixerAddon)
            trainer = trainer_cls(
                model=self.model,
                device=torch.device("cpu"),
                mode="train",
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                precision="auto",
            )
        self.assertIn("requires at least one GPU device", str(context.exception))

    def test_fp32_precision(self):
        """Test FP32 precision with a GPU or CPU."""
        trainer_cls = CustomTrainer.add_on(PrecisionMixerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=self.device,
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision="fp32",
        )
        trainer.run(n_epoch=1, loader=self.loader)
        self.assertEqual(trainer.precision, ["fp32"])

    def test_fp16_precision(self):
        """Test FP16 precision on a GPU-enabled device."""
        if not torch.cuda.is_available():
            self.skipTest("FP16 requires a CUDA-enabled device.")
        trainer_cls = CustomTrainer.add_on(PrecisionMixerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=torch.device("cuda:0"),
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision="fp16",
        )
        trainer.run(n_epoch=1, loader=self.loader)
        self.assertEqual(trainer.precision, ["fp16"])

    def test_auto_precision(self):
        """Test auto precision selection."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for auto precision testing.")
        trainer_cls = CustomTrainer.add_on(PrecisionMixerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=torch.device("cuda:0"),
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision="auto",
        )
        expected_precision = ["bf16", "fp16"] if torch.cuda.get_device_capability(0)[0] >= 8 else ["fp16"]
        self.assertEqual(trainer.precision, expected_precision)

    def test_grad_scaling_with_fp16(self):
        """Test that GradScaler is properly initialized for FP16."""
        if not torch.cuda.is_available():
            self.skipTest("FP16 requires a CUDA-enabled device.")
        trainer_cls = CustomTrainer.add_on(PrecisionMixerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=torch.device("cuda:0"),
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision="fp16",
        )
        trainer.run(n_epoch=1, loader=self.loader)
        self.assertIsNotNone(trainer.scaler, "GradScaler not initialized for FP16 precision.")

    def test_run_batch(self):
        """Test batch processing with auto precision."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for mixed precision training.")
        trainer_cls = CustomTrainer.add_on(PrecisionMixerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=torch.device("cuda:0"),
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision="auto",
        )
        batch = next(iter(self.loader))
        trainer.run_batch(batch)
        self.assertTrue(hasattr(trainer, "log_train"), "Log not updated after running a batch.")


if __name__ == "__main__":
    unittest.main()
