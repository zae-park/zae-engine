# Training Example Using MultiGPUAddon

This document explains how to train a model on multiple GPUs using the MultiGPUAddon. This example extends the Trainer class, previously used for single-GPU training, to perform parallel training across multiple GPUs.

## Requirement

- Python 3.10 or higher
- PyTorch 2.0 or higher
- Two or more CUDA-compatible GPUs

## Script

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torch.distributed as dist
from zae_engine.trainer import Trainer  # Import the custom Trainer class
from zae_engine.trainer.addons.mpu import MultiGPUAddon  # Import MultiGPUAddon
from zae_engine.schedulers import CosineAnnealingScheduler


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()  # Label is 1 if mean > 0.5, otherwise 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define MultiGPUTrainer
class MultiGPUTrainer(Trainer.add_on(MultiGPUAddon)):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

    def test_step(self, batch):
        return self.train_step(batch)

# Usage is the same as before
def main():
    device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]  # Detect all available GPUs
    assert len(device_list) > 1, "This script requires multiple GPUs to run."

    # Dataset and DataLoader
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, optimizer, and scheduler setup
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer, total_iters=100)  # Adjust total_iters

    # Create Trainer instance & start training
    trainer = MultiGPUTrainer(model=model, device=device_list, mode='train', optimizer=optimizer, scheduler=scheduler, init_method="tcp://localhost:12355")
    
    # Execute MultiGPU training
    trainer.run(n_epoch=10, loader=train_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'dummy_model_mgpu.pth')
    print("Model saved as 'dummy_model_mgpu.pth'.")

if __name__ == "__main__":
    main()

```
The script automatically detects and uses all available GPUs for parallel training.


### Explanation

	•	DummyDataset: Generates random data and labels in the range [0, 1].
	•	DummyModel: A simple neural network with a single linear layer for binary classification.
	•	MyTrainer: A custom trainer class that includes forward and backward propagation, loss calculation, and model optimization.
	•	MultiGPUAddon: This addon allows the Trainer class to perform parallel training across multiple GPUs.
	•	Training Process: The model is trained for the specified number of epochs using the Trainer.run method. After training, the model is saved to a file (dummy_model_mgpu.pth).

### Notes

	•	MultiGPUAddon: Requires a system with two or more CUDA-compatible GPUs.
	•	DistributedDataParallel: This example uses PyTorch's DistributedDataParallel to process data in parallel across GPUs and train the model.
	•	init_method: Specifies the method for setting up communication between processes over the network. In this example, localhost is used.

**License**

This project is licensed under the MIT License.