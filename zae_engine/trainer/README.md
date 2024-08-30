# Example of Training Using Trainer

This is an example of how to perform training using a custom Trainer class.

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (Optional: If no GPU is available, the code will run on the CPU)

## Script

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from zae_engine.trainer import Trainer  # Importing the custom Trainer class

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()  # Label 1 if mean > 0.5, else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Custom Trainer Class
class MyTrainer(Trainer):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {'loss': loss}

    def test_step(self, batch):
        return self.train_step(batch)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = DummyDataset(1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, Optimizer, and Scheduler Setup
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = None  # Set a scheduler if needed

    # Create Trainer instance & Train
    trainer = MyTrainer(model=model, device=device, mode='train', optimizer=optimizer, scheduler=scheduler)
    trainer.run(n_epoch=10, loader=train_loader)

    # Save the model
    torch.save(model.state_dict(), 'dummy_model.pth')
    print("The model has been saved as 'dummy_model.pth'.")

if __name__ == "__main__":
    main()
```
The script automatically detects if a GPU is available and uses it for training. If no GPU is available, training will proceed on the CPU.

### Explanation

	•	DummyDataset: Generates random data and labels within the [0, 1] range.
	•	DummyModel: Prepares a simple neural network with a single linear layer for binary classification.
	•	MyTrainer: A custom trainer class that handles forward and backward propagation, loss calculation, and model optimization.
	•	Training Process: Uses the Trainer.run method to train the model for a specified number of epochs. After training, the model is saved to a file (dummy_model.pth).

### Notes

	•	The provided Trainer class is an abstract class, so the train_step and test_step methods must be implemented in a subclass. In this example, MyTrainer implements these methods.
	•	The training script is designed to be easily modified for different datasets, models, and training configurations.

**License**

This project is licensed under the MIT License.