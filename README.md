<div align="center">

<p align="center">
  <img src="assets/img/spinning_ascii_donut.gif" style="border-radius: 20px">
  <br />
</p>

# zae-engine
[ <a href="./README.md">English</a> ]
<span style="display: inline-block; width: 20px;"></span>
[ <a href="./README-ko.md">Korean</a> ]
</div>

<p align="center">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/build_test.yml/badge.svg">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/document_deploy.yml/badge.svg" alt="https://zae-park.github.io/zae-engine">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/unittest_badge.yml/badge.svg">
    <img src="https://github.com/zae-park/zae-engine/actions/workflows/wandb_test.yml/badge.svg">
    </br>
    <img src="https://codecov.io/gh/zae-park/zae-engine/graph/badge.svg?token=4BENXZJHPF">
    <img src="https://img.shields.io/pypi/pyversions/zae-engine.svg" alt="https://pypi.python.org/pypi/zae-engine/">
    <img src="https://img.shields.io/pypi/v/zae-engine.svg" alt="https://pypi.python.org/pypi/zae-engine/">
    <img src="https://img.shields.io/pypi/dm/zae-engine.svg" alt="https://pypi.python.org/pypi/zae-engine/">
  <br />
</p>

# Accelerate AI project !

`zae-engine` is a modular package that consolidates popular deep learning frameworks like PyTorch, TensorFlow, and Flax.
It aims to standardize every step of AI projects, reduce repetitive code, and accelerate data science workflows.
Additionally, it enhances code simplicity and flexibility, making it easier to manage and maintain complex deep learning projects.

## Advantages over unstructured PyTorch
- Improves code readability by organizing commonly used boilerplate code.
- Reduces errors by abstracting away complex engineering tasks handled by `zae-engine`.
- Decouples data science from advanced engineering with minimal code modifications (e.g., CPU, Apple Silicon, CUDA support, and multi-GPU transition).
- Generalizes various implementations for greater flexibility.
- Maintains all flexibility (similar to how `LightningModules` are still PyTorch modules) while removing repetitive boilerplate code.
- Simplifies expansion with easy checkpoint saving, logging, and other extensions.
- Integrates dozens of useful and effective models, layers, and tools.

## Features

| Feature    | Description                                                                                                                                      | Details                                                                                                              |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Data       | Provides `Dataset` for special formats (e.g., `parquet` for large-scale data processing) and `CollateBase` class for modular data preprocessing  | [[README](zae_engine/data/README_dataset.md)]                                                                        |
| Loss       | Provides popular loss functions for angles, size, area, etc.                                                                                     | [[README](zae_engine/loss/README_loss.md)]                                                                           |
| Metrics    | Provides popular evaluation metrics for aggregation, signal, area, etc.                                                                          | [[README](zae_engine/metrics/README_metrics.md)]                                                                     |
| Models     | Provides builders for various deep learning models, implementation of popular models, and `dim_converter` for model dimensionality conversion    | [[README](zae_engine/models/README_models.md)]                                                                       |
| NN_night   | Provides various deep learning layers and blocks                                                                                                 | [[README-blocks](zae_engine/nn_night/README_blocks.md)] <br> [[README-layers](zae_engine/nn_night/README_layers.md)] |
| Operation  | Provides post-processing and transformations based on algorithms                                                                                 | [[README](zae_engine/operation/README_operation.md)]                                                                 |
| Schedulers | Provides modular learning rate management with `SchedulerBase` and popular learning rate schedulers                                              | [[README](zae_engine/schedulers/README_schedulers.md)]                                                               |
| Trainer    | Provides `Trainer` to manage training instances and `add-on` features for advanced engineering such as state management and distributed training | [[README](zae_engine/trainer/README_trainer.md)]                                                                     |

### Support Frameworks (WIP)
- [x] PyTorch
- [ ] TensorFlow
- [ ] Flax

### zae-command
After installation, the `zae` command can be used in the terminal to access the following features:
- `zae hello`: Verify successful installation
- `zae example`: Create an example script file ***zae_example.py***
- `zae tree`: Print the supported classes and functions within the package

## Installation

### PIP
```bash
pip install zae-engine
# or
git+https://github.com/zae-park/zae-engine.git
```

### Poetry
```bash
poetry add zae-engine
# or
git+https://github.com/zae-park/zae-engine.git
```

## LICENSE
This project is licensed under the [Apache License 2.0](./LICENSE).

