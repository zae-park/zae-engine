# Trainer Sub-Package

The Trainer sub-package of the `zae-engine` is designed to facilitate and manage the training and evaluation process of deep learning models. It provides an abstract base class for defining custom training workflows and also includes several add-ons to extend the core functionality. The flexibility of the Trainer class allows users to customize their training, testing, and validation steps, making it suitable for a wide range of machine learning experiments.

## Core Components

### `_trainer.py`
The main file in this sub-package contains the core class `Trainer`, an abstract base class for managing training and evaluation.

#### `Trainer` Class
The `Trainer` class is an abstract class that requires users to define `train_step` and `test_step` methods for training and testing logic, respectively. This class has several features:

- **Model Management**: Handles model transfer between devices (e.g., CPU and GPU).
- **Training Loop**: Manages the training loop, including data loading, logging, and scheduler updates.
- **Logging**: Logs training and evaluation metrics for each batch and epoch.
- **Add-ons**: The trainer class can be extended with additional functionalities using add-ons.

Key features include:
- **Gradient Clipping**: Clips gradients to prevent exploding gradients.
- **Learning Rate Scheduler**: Supports learning rate scheduling.
- **Device Management**: Handles both single and multi-GPU setups.
- **Logging and Progress Tracking**: Provides support for logging metrics and showing training progress.
- **Model Saving and Loading**: Saves model states and allows loading for resuming training.

The `Trainer` class provides a flexible structure for managing machine learning experiments, offering customization points for the user to define their training and evaluation logic.

#### `ProgressChecker` Class
This helper class is used to track the progress of training and evaluation, such as counting steps and epochs.

### Add-ons
The Trainer sub-package includes an add-ons feature that allows extending the functionality of the core `Trainer` class with minimal effort. This is useful for adding common features without modifying the core trainer code.

#### `AddOnBase` Class (in `addons/core.py`)
This is the base class for creating add-ons that can extend the functionality of the Trainer. Add-ons can be installed by calling `Trainer.add_on()` and passing in one or more add-on classes.

#### State Management (`addons/state_manager.py`)
Provides state saving and loading capabilities. It allows users to save model, optimizer, and scheduler states, and then load them to resume training seamlessly. The state can be saved in different formats such as checkpoint or `safetensors`.

#### Web Logger (`addons/web_logger.py`)
Provides integration with popular logging services such as WandB and Neptune to facilitate remote experiment tracking. These services allow you to monitor and visualize metrics during training.

#### Multi-GPU Training (`addons/mpu.py`)
Adds multi-GPU support using PyTorch's DistributedDataParallel (DDP). It allows the training process to be distributed across multiple GPUs to improve training speed and handle large models.

## Usage
To create a custom trainer, inherit from the `Trainer` class and implement the `train_step` and `test_step` methods:

```python
from zae_engine.trainer import Trainer

class CustomTrainer(Trainer):
    def train_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        return {"loss": loss}

    def test_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        return {"loss": loss}
```

To add state management and multi-GPU support:

```python
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import StateManagerAddon, MultiGPUAddon

ExtendedTrainer = Trainer.add_on(StateManagerAddon, MultiGPUAddon)
trainer = ExtendedTrainer(model, device, mode='train', optimizer=optimizer, scheduler=scheduler, save_path='checkpoints/')
```

## Summary
The Trainer sub-package in `zae-engine` provides a flexible and extensible solution for managing machine learning experiments. With the core `Trainer` class, users can easily create customized training workflows. The add-ons feature allows for extending functionality with minimal code, providing a modular way to enhance training capabilities.

