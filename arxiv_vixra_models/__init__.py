# Plotting
import matplotlib.pyplot as plt
# ML imports.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, AveragePrecision, StatScores
import pandas as pd
import wandb
# Requires sklearn v0.24 or greaters. Must upgrade in Colab.
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
# Type Hinting.
from typing import Dict, List, Tuple, Sequence, Optional, TypeVar, Union, Callable
from typing_extensions import Literal
# Type-hint aliasing
Num = TypeVar('Num', 'float', 'int')
Tensor = Sequence[Num]
DF = pd.DataFrame
# For text normalization
from unidecode import unidecode
import unicodedata
import string
# Misc
import math
import json

from .models import LitOneHotCharRecurrent, LitEmbeddingRecurrent
from .datamodules import OneHotCharDataModule, EmbeddingDataModule
from .callbacks import WandbTextCallback
from .text_normalizer import text_normalizer
from .one_hot import one_hot_encoding, one_hot_decoding
from .embedding import string_to_ints, ints_to_string

__all__ = ['LitOneHotCharRecurrent',
            'LitEmbeddingRecurrent',
           'OneHotCharDataModule',
           'EmbeddingDataModule',
           'WandbTextCallback',
           'text_normalizer',
           'one_hot_encoding',
           'one_hot_decoding',
           'string_to_ints',
           'ints_to_string']