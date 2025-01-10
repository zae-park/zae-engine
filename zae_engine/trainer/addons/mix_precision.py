from typing import Type, Union

import torch
from torch.cuda.amp import autocast, GradScaler

from .core import AddOnBase
from .._trainer import T


class PrecisionMixerAddon(AddOnBase):
    """
    Add-on for mixed precision training.

    This add-on enables mixed precision training to improve computational efficiency and reduce memory usage.
    It supports automatic precision selection or user-defined precision settings (e.g., 'fp32', 'fp16', 'bf16').

    Parameters
    ----------
    precision : Union[str, list], optional
        The precision setting for training. Default is "auto".
        - "auto": Automatically selects the best precision based on hardware capabilities.
        - "fp32": Uses full precision (default in PyTorch).
        - "fp16": Uses half precision for accelerated computation.
        - "bf16": Uses Brain Float 16 precision for supported hardware.
        - List: Specifies a priority order for precision (e.g., ["bf16", "fp16"]).

    Methods
    -------
    run_batch(batch, **kwargs)
        Override the batch processing method to apply mixed precision.

    Notes
    -----
    - Mixed precision improves training speed and reduces memory usage by performing certain operations
      in lower precision (e.g., FP16) while maintaining stability in others (e.g., FP32 for loss calculation).
    - Automatically handles loss scaling via `torch.cuda.amp.GradScaler` to prevent overflow issues
      when using FP16.

    Examples
    --------
    Using PrecisionMixerAddon with auto precision:

    >>> from zae_engine.trainer import Trainer
    >>> from zae_engine.trainer.addons import PrecisionMixerAddon

    >>> MyTrainer = Trainer.add_on(PrecisionMixerAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     precision='auto'  # Automatically selects best precision
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader, valid_loader=valid_loader)

    Using a priority list for precision:

    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     precision=["bf16", "fp16"]  # Tries bf16 first, falls back to fp16
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader)
    """

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        class PrecisionTrainer(base_cls):
            def __init__(self, *args, precision: Union[str, list] = "auto", **kwargs):
                super().__init__(*args, **kwargs)

                # Check if all devices are CPU
                if all("cpu" in str(device) for device in self.device):
                    raise ValueError("PrecisionMixerAddon requires at least one GPU device to function.")

                # Determine precision settings
                self.precision = self._determine_precision(precision)
                self.scaler = GradScaler() if "fp16" in self.precision else None

            def _determine_precision(self, precision):
                """
                Determine the appropriate precision settings based on user input or hardware capabilities.

                Parameters
                ----------
                precision : Union[str, list]
                    The user-specified precision setting.
                    - "auto": Automatically selects the best precision based on the GPU's hardware capabilities.
                    - str: A single precision mode (e.g., "fp32", "fp16", "bf16").
                    - list: A priority order for precision modes (e.g., ["bf16", "fp16"]).

                Returns
                -------
                list
                    A list of supported precision modes, ordered by priority.

                Raises
                ------
                ValueError
                    If the `precision` argument is not a valid string or list.

                Notes
                -----
                - When "auto" is specified, the method checks the hardware's compute capability to determine
                  the best available precision (e.g., "bf16" on NVIDIA Ampere GPUs, otherwise "fp16").
                - If multiple precision modes are specified in a list, the method validates their compatibility
                  with the hardware and prioritizes them as given.

                Examples
                --------
                Automatically select the best precision:

                >>> addon._determine_precision("auto")
                ['bf16', 'fp16']

                Use a specific precision:

                >>> addon._determine_precision("fp32")
                ['fp32']

                Provide a priority list:

                >>> addon._determine_precision(["bf16", "fp16"])
                ['bf16', 'fp16']
                """
                if precision == "auto":
                    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
                        return ["bf16", "fp16"]  # Take BF16 first @ over `Ampere`
                    elif torch.cuda.is_available():
                        return ["fp16"]
                    else:
                        return ["fp32"]
                elif isinstance(precision, str):
                    return [precision]
                elif isinstance(precision, list):
                    return precision
                else:
                    raise ValueError("Invalid precision setting.")

            def run_batch(self, batch, **kwargs):
                """
                Override the batch processing method to enable mixed precision training.

                Parameters
                ----------
                batch : Union[tuple, dict]
                    The batch of data to process.
                kwargs : dict
                    Additional arguments for batch processing.

                Notes
                -----
                - If precision is set to "fp16" or "bf16", this method uses `torch.cuda.amp.autocast`
                  to enable mixed precision operations for the forward pass.
                - Loss scaling is automatically applied when using FP16 to prevent overflow issues.
                - The precision setting does not affect the input data's original precision; it only
                  applies during model computations and optimizer steps.

                Examples
                --------
                Forward pass and gradient scaling with mixed precision:

                >>> trainer.run_batch(batch)
                >>> # AutoCast ensures model operations are performed in fp16 or bf16
                """
                batch = self._to_device(**batch) if isinstance(batch, dict) else self._to_device(*batch)

                if self.mode == "train":
                    self.model.train()
                    self.optimizer.zero_grad()

                    with autocast(enabled="fp16" in self.precision or "bf16" in self.precision, dtype=torch.float16):
                        step_dict = self.train_step(batch)
                        loss = step_dict["loss"]

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    if self.scheduler_step_on_batch:
                        self.scheduler.step(**kwargs)

                elif self.mode == "test":
                    self.model.eval()
                    with torch.no_grad():
                        with autocast(
                            enabled="fp16" in self.precision or "bf16" in self.precision, dtype=torch.float16
                        ):
                            step_dict = self.test_step(batch)
                else:
                    raise ValueError(f"Unexpected mode {self.mode}.")

                torch.cuda.synchronize()
                self.logging(step_dict)
                self.progress_checker.update_step()

        return PrecisionTrainer
