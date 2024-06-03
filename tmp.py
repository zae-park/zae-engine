import ssl
import datetime
from collections import namedtuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from zae_engine import trainer, models, metrics
from zae_engine.models.foundations import unet, resnet
from zae_engine.nn_night.blocks import UNetBlock
from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.metrics.confusion import confusion_matrix, print_confusion_matrix


if __name__ == "__main__":
    dummy_x = torch.zeros((1, 3, 256, 256))
    model = unet.unet(pretrained=True)
    out = model(dummy_x)

    print()
