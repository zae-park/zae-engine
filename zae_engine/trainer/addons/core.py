from typing import Type
from abc import ABC, abstractmethod

from .._trainer import T


class AddOnBase(ABC):
    """
    Base class for defining add-ons for the Trainer class.

    Add-ons allow you to extend the functionality of the Trainer class dynamically.
    By inheriting from `AddOnBase`, subclasses can implement custom functionality
    that can be integrated into the Trainer workflow.

    Methods
    -------
    apply(cls, base_cls)
        Apply the add-on to the specified base class, modifying its behavior or adding
        new features. This method must be implemented by subclasses.

    Notes
    -----
    - Add-ons are designed to be composable, meaning you can apply multiple add-ons
      to a single Trainer class.
    - Subclasses must override the `apply` method to specify how the add-on modifies
      the behavior of the Trainer class.

    Examples
    --------
    Creating a custom add-on:

    >>> from zae_engine.trainer.addons import AddOnBase

    >>> class CustomLoggerAddon(AddOnBase):
    >>>     @classmethod
    >>>     def apply(cls, base_cls):
    >>>         class TrainerWithCustomLogger(base_cls):
    >>>             def __init__(self, *args, custom_param=None, **kwargs):
    >>>                 super().__init__(*args, **kwargs)
    >>>                 self.custom_param = custom_param
    >>>             def logging(self, step_dict):
    >>>                 super().logging(step_dict)
    >>>                 print(f"Custom logging: {step_dict}")
    >>>         return TrainerWithCustomLogger

    Applying the custom add-on to a Trainer:

    >>> from zae_engine.trainer import Trainer
    >>> MyTrainer = Trainer.add_on(CustomLoggerAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     custom_param="Log everything"
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader)
    """

    @classmethod
    @abstractmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        """
        Apply the add-on modifications to a base class.

        This method is used to inject additional functionality or modify the behavior of the
        `Trainer` class. It returns a new class that combines the original `Trainer` with
        the behavior defined in the add-on.

        Parameters
        ----------
        base_cls : Type[T]
            The base class to which the add-on modifications will be applied. This is typically
            the `Trainer` class or a subclass of it.

        Returns
        -------
        Type[T]
            A new class that includes the functionality of the base class along with the
            additional behavior defined by the add-on.

        Notes
        -----
        - Subclasses of `AddOnBase` must implement this method to define how the add-on modifies
          the base class.
        - This method is typically called indirectly through the `Trainer.add_on` method.

        Examples
        --------
        Custom implementation in an add-on:

        >>> class CustomAddon(AddOnBase):
        >>>     @classmethod
        >>>     def apply(cls, base_cls):
        >>>         class TrainerWithCustomAddon(base_cls):
        >>>             def custom_method(self):
        >>>                 print("Custom method called")
        >>>         return TrainerWithCustomAddon

        Adding the custom add-on to a Trainer:

        >>> from zae_engine.trainer import Trainer
        >>> MyTrainer = Trainer.add_on(CustomAddon)
        >>> trainer = MyTrainer(model=my_model, device='cuda', optimizer=my_optimizer)
        >>> trainer.custom_method()  # Output: Custom method called
        """
        raise NotImplementedError("Add-on must implement the apply method.")
