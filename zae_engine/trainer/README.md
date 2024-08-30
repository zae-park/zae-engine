# Single GPU Training Example with Dummy Data

This repository contains an example of how to perform training on a single GPU using a custom `Trainer` class.

## Requirements

- Python 3.x
- PyTorch 1.9 or later
- CUDA compatible GPU (optional, the code will run on CPU if GPU is not available)

## Example Components

- **DummyDataset**: A simple dataset that generates random data and assigns labels based on a threshold.
- **DummyModel**: A basic linear model for binary classification.
- **MyTrainer**: A custom trainer class that handles training and evaluation steps.
- **single_gpu_training.py**: The script responsible for setting up the training process on a single GPU.

## How to Run

1. Clone this repository and navigate to the project directory.
2. Ensure that you have the necessary dependencies installed.
3. Run the training script:

   ```bash
   python single_gpu_training.py
   ```

The script will automatically detect if a GPU is available and use it for training. If a GPU is not available, it will fall back to using the CPU.

Explanation

	•	DummyDataset: Generates random data in the range [0, 1] and assigns labels based on whether the data points fall within a specific range.
	•	DummyModel: A simple neural network with a single linear layer for binary classification.
	•	MyTrainer: A custom trainer class that handles the training loop, including forward and backward passes, loss calculation, and model optimization.
	•	Training Process: The Trainer.run method is used to train the model for a specified number of epochs. After training, the model is saved to a file (dummy_model.pth).

Notes

	•	The provided Trainer class is an abstract class, meaning that train_step and test_step methods must be implemented by subclasses. In this example, MyTrainer implements these methods.
	•	The single_gpu_training.py script is designed to be straightforward and easy to modify for different datasets, models, and training configurations.

License

This project is licensed under the MIT License.