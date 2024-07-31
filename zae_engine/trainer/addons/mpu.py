import torch

from .core import AddOnBase


class MultiGPUAddon(AddOnBase):
    """
    Add-on for enabling multi-GPU support in the Trainer class.

    Methods
    -------
    apply(cls, base_cls)
        Applies the multi-GPU modifications to the base class.
    """

    @classmethod
    def apply(cls, base_cls):
        """
        Applies the multi-GPU modifications to the base class.

        Parameters
        ----------
        base_cls : Type
            The base class to which the multi-GPU modifications will be applied.

        Returns
        -------
        Type
            The modified base class with multi-GPU support.
        """

        class MultiGPUTrainer(base_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if isinstance(self.device, list):
                    self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
                    self.device = torch.device(f"cuda:{self.device[0]}")
                    self.model.to(self.device)

        return MultiGPUTrainer
