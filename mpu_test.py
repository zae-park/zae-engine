import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)  # Data in the range [0, 1]
        self.labels = (
            (self.data.mean(dim=1) > 0.4) & (self.data.mean(dim=1) < 0.6)
        ).long()  # Label is 1 if mean is in (0.4, 0.6)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def train(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12355", world_size=world_size, rank=rank)

    # Set the device
    torch.cuda.set_device(rank)

    # Create the dataset and dataloader
    dataset = DummyDataset(size=10000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Initialize the model, loss function, and optimizer
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # Small learning rate for gradual learning

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Ensures different data shuffling each epoch
        total_loss = 0.0
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(rank), labels.to(rank)

            # Forward pass
            outputs = ddp_model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # Cleanup
    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()  # Number of GPUs available
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
