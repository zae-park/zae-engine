# Trainer Sub-package Overview

The Trainer sub-package of the **zae-engine** library provides essential features for managing model training and extending functionality through a variety of add-ons. The `Trainer` class is the core component that oversees the entire training process, offering abstract methods for implementing custom training and testing steps. With the add-on system, users can easily enhance and customize their training workflows.

## Key Classes and Features

### Trainer

`Trainer` is an abstract base class designed for managing the training routines of a model. It requires users to subclass it and implement the `train_step` and `test_step` methods, which define the behavior during training and testing, respectively.

#### Main Attributes
- **model**: The model to be trained (`torch.nn.Module`).
- **device**: Device(s) for training the model (e.g., 'cuda' or ['cuda:0', 'cuda:1']).
- **optimizer**: Optimizer used for training (`torch.optim.Optimizer`).
- **scheduler**: Learning rate scheduler (`torch.optim.lr_scheduler._LRScheduler`).
- **log_bar**: Whether to display a progress bar during training.
- **gradient_clip**: Value for gradient clipping (default: 0.0).

#### Main Methods
- **train_step(batch)**: Defines the operations for each training step.
- **test_step(batch)**: Defines the operations for each testing step.
- **run(n_epoch, loader, valid_loader)**: Runs the training or testing for the specified number of epochs.
- **run_epoch(loader)**: Runs training or testing for a single epoch.
- **run_batch(batch)**: Runs training or testing for a single batch.
- **add_on(*add_on_cls)**: Adds functionality to the `Trainer` class via add-ons.

### ProgressChecker

`ProgressChecker` is a helper class for tracking training or testing progress. It manages epoch and step counts, allowing users to easily monitor the status of their training process.

## Add-on Features
The Trainer sub-package supports various add-ons via the `AddOnBase` class, which allows for extending functionality such as distributed training, state management, and web logging.

### Main Add-ons

- **StateManagerAddon**: Provides functionality to save and load checkpoints during model training, including the model state, optimizer state, and scheduler state.
- **WandBLoggerAddon / NeptuneLoggerAddon**: Enables real-time monitoring of the training process using Weights & Biases or Neptune.
- **MultiGPUAddon**: Adds multi-GPU support for distributed training using DDP (Distributed Data Parallel), allowing models to be trained across multiple GPUs in parallel.

## Usage Example
The Trainer sub-package makes it easy to manage and extend model training in various scenarios. Users can inherit from the `Trainer` class, implement their custom `train_step` and `test_step` methods, and add necessary add-ons to set up their training environment.

```python
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import StateManagerAddon, WandBLoggerAddon

# Add StateManager and WandBLogger add-ons to Trainer
MyTrainer = Trainer.add_on(StateManagerAddon, WandBLoggerAddon)

trainer = MyTrainer(
    model=my_model,
    device='cuda',
    mode='train',
    optimizer=my_optimizer,
    scheduler=my_scheduler,
    save_path='./checkpoints',
    web_logger={"wandb": {"project": "my_project"}},
)

trainer.run(n_epoch=10, loader=train_loader, valid_loader=valid_loader)
```
This example shows how to extend the `Trainer` class with state management and web logging capabilities, allowing users to effectively monitor the model training process and manage checkpoints.

