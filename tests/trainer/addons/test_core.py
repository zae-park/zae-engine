import unittest
from typing import Type

import torch

from zae_engine.trainer import Trainer
from zae_engine.trainer.addons.core import AddOnBase


# Define a dummy add-on for testing
class DummyAddon(AddOnBase):
    @classmethod
    def apply(cls, base_cls: Type[Trainer]) -> Type[Trainer]:
        class TrainerWithDummyAddon(base_cls):
            def dummy_method(self):
                return "Dummy method executed."

        return TrainerWithDummyAddon


# Define a dummy Trainer for testing
class DummyTrainer(Trainer):
    def train_step(self, batch):
        return {"loss": 0.0}

    def test_step(self, batch):
        return {"loss": 0.0}


class TestCore(unittest.TestCase):
    def test_addon_integration(self):
        """Test that an AddOnBase subclass can be integrated with the Trainer."""
        TrainerWithAddon = DummyTrainer.add_on(DummyAddon)
        trainer = TrainerWithAddon(
            model=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            mode="train",
            optimizer=None,
            scheduler=None,
        )
        self.assertTrue(hasattr(trainer, "dummy_method"), "Add-on method not integrated.")
        self.assertEqual(trainer.dummy_method(), "Dummy method executed.", "Add-on method did not execute correctly.")

    def test_addon_abstract_method(self):
        """Test that AddOnBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            AddOnBase()  # Attempt to instantiate abstract class

    def test_multiple_addons(self):
        """Test that multiple AddOns can be applied to a Trainer."""

        class AnotherDummyAddon(AddOnBase):
            @classmethod
            def apply(cls, base_cls: Type[Trainer]) -> Type[Trainer]:
                class TrainerWithAnotherAddon(base_cls):
                    def another_dummy_method(self):
                        return "Another dummy method executed."

                return TrainerWithAnotherAddon

        TrainerWithMultipleAddons = DummyTrainer.add_on(DummyAddon, AnotherDummyAddon)
        trainer = TrainerWithMultipleAddons(
            model=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            mode="train",
            optimizer=None,
            scheduler=None,
        )
        self.assertTrue(hasattr(trainer, "dummy_method"), "First add-on method not integrated.")
        self.assertTrue(hasattr(trainer, "another_dummy_method"), "Second add-on method not integrated.")
        self.assertEqual(
            trainer.dummy_method(), "Dummy method executed.", "First add-on method did not execute correctly."
        )
        self.assertEqual(
            trainer.another_dummy_method(),
            "Another dummy method executed.",
            "Second add-on method did not execute correctly.",
        )


if __name__ == "__main__":
    unittest.main()
