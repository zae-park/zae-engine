# Scheduler Sub-package

This sub-package provides implementations for learning rate schedulers, which are essential tools to adjust the learning rate during training for optimal performance. The schedulers included in this package are designed to provide flexible and efficient learning rate management throughout the training process. This README introduces the provided scheduler classes and their functionalities.

### Overview of Classes

1. **SchedulerBase** (Abstract Base Class)
   - `SchedulerBase` is an abstract base class that extends PyTorch's `LRScheduler`. Users can inherit from this class and implement their custom learning rate schedulers by defining the `get_lr` method.
   - **Parameters**:
     - `optimizer`: The optimizer for which the learning rate will be scheduled (e.g., Adam, SGD).
     - `total_iters`: The total number of iterations for the scheduler.
     - `eta_min`: The minimum learning rate value.
     - `last_epoch`: The index of the last epoch (default is -1).

2. **SchedulerChain**
   - `SchedulerChain` allows chaining multiple schedulers in sequence for learning rate management over extended training epochs. When the current scheduler completes, the next scheduler in the chain takes over.
   - **Parameters**:
     - `*schedulers`: A variable number of schedulers to be chained together, each of which is a subclass of `SchedulerBase`.
   - This allows users to combine different learning rate schedules effectively during different phases of training.

3. **WarmUpScheduler** (Derived from `SchedulerBase`)
   - The `WarmUpScheduler` is a linear warm-up scheduler that increases the learning rate from a minimum (`eta_min`) to the base learning rate over the specified number of iterations (`total_iters`).
   - **Parameters**:
     - `optimizer`: The optimizer for which the learning rate will be scheduled.
     - `total_iters`: The number of iterations over which the learning rate will warm up.
     - `eta_min`: The starting minimum learning rate value.
     - `last_epoch`: The index of the last epoch (default is -1).
   - **Usage**: This scheduler is typically used in the early training phase to prevent sudden updates to model parameters by gradually increasing the learning rate.

4. **CosineAnnealingScheduler** (Derived from `SchedulerBase`)
   - `CosineAnnealingScheduler` reduces the learning rate following a cosine annealing curve, starting from the base learning rate and reducing it to a minimum (`eta_min`) over the specified number of iterations (`total_iters`).
   - **Parameters**:
     - `optimizer`: The optimizer for which the learning rate will be scheduled.
     - `total_iters`: The number of iterations over which the learning rate will anneal.
     - `eta_min`: The minimum learning rate value.
     - `last_epoch`: The index of the last epoch (default is -1).
   - **Usage**: This scheduler is ideal for gradually reducing the learning rate as the model converges.

### Example Usage

Below is an example demonstrating how to use the schedulers from this package.

```python
import torch
from torch.optim import Adam
from zae_engine.schedulers import WarmUpScheduler, CosineAnnealingScheduler, SchedulerChain

# Example model and optimizer
model = torch.nn.Linear(10, 1)
optimizer = Adam(model.parameters(), lr=0.001)

# Initialize schedulers
warmup_scheduler = WarmUpScheduler(optimizer, total_iters=1000, eta_min=0.0001)
cosine_scheduler = CosineAnnealingScheduler(optimizer, total_iters=5000, eta_min=0.00001)

# Chain schedulers together
scheduler_chain = SchedulerChain(warmup_scheduler, cosine_scheduler)

# Training loop
for epoch in range(10000):
    # Training code here...
    # Step the scheduler to update the learning rate
    scheduler_chain.step()
    current_lr = scheduler_chain.get_lr()
    print(f"Epoch {epoch}, Learning Rate: {current_lr}")
```

In this example, the learning rate starts with a warm-up phase (`WarmUpScheduler`) for the first 1000 iterations and then switches to a cosine annealing schedule (`CosineAnnealingScheduler`) for the remaining 5000 iterations, giving a smooth learning rate transition.

### Summary
The scheduler sub-package provides the following key features:
- **Custom Learning Rate Schedulers**: Easily implement custom learning rate schedules by inheriting from `SchedulerBase`.
- **Chaining Schedulers**: Chain multiple schedulers together using `SchedulerChain` to create complex learning rate schedules for long training processes.
- **Provided Schedulers**: Use ready-made schedulers such as `WarmUpScheduler` and `CosineAnnealingScheduler` to manage learning rate changes effectively.

These schedulers are designed to be flexible and easy to integrate into PyTorch training workflows, providing the ability to handle different training phases optimally.

