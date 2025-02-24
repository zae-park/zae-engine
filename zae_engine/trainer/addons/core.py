from typing import Type, Optional, Dict, Tuple
from abc import ABC, abstractmethod

from .._trainer import T


class AddOnBase(ABC):
    """
    Base class for defining add-ons for the Trainer class.

    Add-ons allow you to extend the functionality of the Trainer class dynamically.
    By inheriting from `AddOnBase`, subclasses can implement custom functionality
    that can be integrated into the Trainer workflow.

    Attributes
    ----------
    addon_case : int
        Defines the type of Add-on (0-3).

    Methods
    -------
    apply(cls, base_cls)
        Apply the add-on to the specified base class, modifying its behavior or adding
        new features. This method must be implemented by subclasses.

    core(self, data: Optional[dict]) -> Tuple[int, Optional[dict]]
        Execute the add-on logic during the Trainer workflow.

    Notes
    -----
    - Add-ons are designed to be composable, meaning you can apply multiple add-ons
      to a single Trainer class.
    - Subclasses must override the `apply` method to specify how the add-on modifies
      the behavior of the Trainer class.
    - The `core()` method is responsible for executing the main functionality of an add-on
      and returning a status code along with optional output data.

    Examples
    --------
    Creating a custom add-on:

    >>> from zae_engine.trainer.addons import AddOnBase

    >>> class CustomLoggerAddon(AddOnBase):
    >>>     addon_case = 3  # This add-on requires input data
    >>>     def core(self, data):
    >>>         print(f"Custom logging: {data}")
    >>>         return 0, None

    Applying the custom add-on to a Trainer:

    >>> from zae_engine.trainer import Trainer
    >>> MyTrainer = Trainer.add_on(CustomLoggerAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader)
    """

    addon_case: int = None  # This must be defined in subclasses

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
        """
        raise NotImplementedError("Add-on must implement the apply method.")

    def core(self, data: Optional[Dict] = None) -> Tuple[int, Optional[Dict]]:
        """
        Core execution logic for the Add-on.

        Parameters
        ----------
        data : dict, optional
            Input data passed from the previous Add-on.

        Returns
        -------
        status : int
            -1: Failure, stops execution
             0: Successful execution, no data transfer
             1: Successful execution, with data transfer
             2: Trigger event (next Add-on required)
        output : dict, optional
            The output data to be passed to the next Add-on, if applicable.

        Notes
        -----
        - The default implementation does nothing and simply returns (0, None).
        - Add-ons that require data must override this method and define their behavior.
        """
        return 0, None  # Default behavior: no effect, no data transfer
