아래와 같이 Trainer 서브패키지의 README에 MultiGPUAddon에 대한 내용을 추가하여, advanced 섹션으로 포함시켰습니다. MultiGPUAddon 사용 방법에 대한 예제와 설명을 포함하여 전체적으로 사용자가 쉽게 접근할 수 있도록 작성했습니다.

---

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

## Advanced: MultiGPUAddon

The `MultiGPUAddon` provides a powerful way to train models in parallel across multiple GPUs, significantly speeding up the training process for large-scale models. This add-on leverages PyTorch's Distributed Data Parallel (DDP) to ensure that models and data are properly synchronized across all GPUs.

### Example: Using MultiGPUAddon for Training

Below is an example of how to use the `MultiGPUAddon` to train a model across multiple GPUs:

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons.mpu import MultiGPUAddon
from zae_engine.schedulers import CosineAnnealingScheduler

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# MultiGPUTrainer Definition
class MultiGPUTrainer(Trainer.add_on(MultiGPUAddon)):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

    def test_step(self, batch):
        return self.train_step(batch)

# Main function
def main():
    device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    assert len(device_list) > 1, "This script must be run with multiple GPUs available."

    # Dataset and DataLoader
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, Optimizer, and Scheduler Setup
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer, total_iters=100)

    # Create Trainer Instance & Train
    trainer = MultiGPUTrainer(
        model=model,
        device=device_list,
        mode='train',
        optimizer=optimizer,
        scheduler=scheduler,
        init_method="tcp://localhost:12355"
    )
    
    trainer.run(n_epoch=10, loader=train_loader)
    torch.save(model.state_dict(), 'dummy_model_mgpu.pth')
    print("Model saved as 'dummy_model_mgpu.pth'.")

if __name__ == "__main__":
    main()
```

### Explanation
- **DummyDataset**: Generates random data in the range [0, 1] and corresponding labels.
- **DummyModel**: A simple single-layer neural network for binary classification.
- **MultiGPUTrainer**: A custom trainer class that inherits from `Trainer` and uses `MultiGPUAddon` to train across multiple GPUs.
- **Training Process**: The `Trainer.run` method is used to train the model for the specified number of epochs. After training, the model is saved to a file (`dummy_model_mgpu.pth`).

### Important Notes
- **CUDA-compatible GPUs**: At least two CUDA-compatible GPUs are required to use `MultiGPUAddon`.
- **Distributed Data Parallel**: This example utilizes PyTorch's `DistributedDataParallel` to parallelize data processing and model training across all GPUs.
- **init_method**: Specifies how to initialize communication across processes. In this example, we use `localhost` to set up the communication.

`MultiGPUAddon` is an advanced add-on that requires familiarity with distributed training and GPU management. It provides the tools necessary for efficiently training large models in parallel, significantly reducing the time required to train models on large datasets.