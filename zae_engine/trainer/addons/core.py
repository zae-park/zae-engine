from abc import ABC, abstractmethod


class AddOnBase:
    """
    Base class for add-ons that can be installed on the Trainer class.

    Methods
    -------
    apply(cls, base_cls)
        Applies the add-on modifications to the base class.
    """

    @classmethod
    def apply(cls, base_cls):
        """
        Applies the add-on modifications to the base class.

        Parameters
        ----------
        base_cls : Type
            The base class to which the add-on modifications will be applied.
        """
        raise NotImplementedError("Add-on must implement the apply method.")
