Welcome to ZAE-Engine's Documentation!
=======================================

ZAE-Engine is a modular AI framework designed to streamline and accelerate AI workflows.
Currently, it supports PyTorch and provides tools for model training, evaluation, and deployment.

.. image:: https://img.shields.io/pypi/v/zae-engine.svg
    :alt: PyPI version
    :align: center

.. note::
   This documentation is for the **pre-release** version of ZAE-Engine and is subject to change.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents

   installation
   usage
   modules/core
   modules/addons
   api_reference

Quick Links
===========

- :doc:`installation`
- :doc:`usage`
- :doc:`modules/core`
- :doc:`api_reference`

Getting Started
===============

If you're new to ZAE-Engine, we recommend starting with the **Installation** and **Usage** sections.

Key Features
------------

- Simplifies repetitive coding tasks with modular utilities.
- Supports PyTorch for flexible AI workflows.
- Provides easy integration of add-ons for state management, distributed training, and logging.

Command Line Interface
----------------------

After installing ZAE-Engine, you can use the `zae` command to simplify various tasks:

- **`zae hello`**: Verifies that the installation was successful.
- **`zae example`**: Generates an example script (`zae_example.py`) for quick reference.
- **`zae tree`**: Displays the available classes and functions in the package.

Quick Start
-----------

Here's a quick example of using the `Trainer` class for model training:

.. code-block:: python

   from zae_engine.trainer import Trainer
   from torch.optim import Adam
   from torch.nn import Linear

   model = Linear(10, 2)  # Example model
   trainer = Trainer(
       model=model,
       optimizer=Adam(model.parameters(), lr=0.001),
       device='cuda'
   )
   trainer.run(n_epoch=10, loader=train_loader, valid_loader=valid_loader)

For more detailed examples, check out the **Usage** section.

Modules
=======

Explore the core functionalities of ZAE-Engine through its modules:

.. toctree::
   :maxdepth: 4
   :caption: Modules:

   data/zae_engine.data.rst
   loss/zae_engine.loss.rst
   metrics/zae_engine.metrics.rst
   models/zae_engine.models.rst
   nn_night/zae_engine.nn_night.rst
   operation/zae_engine.operation.rst
   schedulers/zae_engine.schedulers.rst
   trainer/zae_engine.trainer.rst
   utils/zae_engine.utils.rst

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
