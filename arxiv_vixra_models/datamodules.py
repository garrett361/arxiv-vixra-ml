import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Union

from .datasets import OneHotCharDataset, EmbeddingDataset, OneHotCharCheatingDataset, EmbeddingCheatingDataset


class OneHotCharDataModule(pl.LightningDataModule):
    """Simple pl.LightningDataModule wrapper for torch one-hot-text datasets.

    Description
    ----------
    Data and the 1-indexed char dict are expected to be in feather format.
    The `setup` method initializes the appropriate Dataset objects, which are
    assumed to be returning one-hot encoded text via their `__getitem__`
    methods.

    Args
    ----------
    `train_data`: pd.DataFrame or str
        DataFrame object or path to training data feather file.
    `val_data`: pd.DataFrame or str
        DataFrame object or path to training data feather file.
    `tokens`: pd.DataFrame or str
        DataFrame object or path to character feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        How many characters are read in from the text.
    `batch_size`: int, default = 1
        Dataloader batch size.
    `num_workers`: int, default = 1
        Dataloader's number of workers used.
    `pin_memory`: bool, default = False
        Dataloader's pin_memory option.
    `persistent_workers`: bool, default = False
        Dataloader's persistent_workers option.
    `sample_size`: int or float or None, optional, default = None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = True
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods
    ----------
    `test_dataloader`
        Not-yet implemented as to protect against accidentally viewing test set.
     """

    def __init__(self,
                 train_data: Union[str, pd.DataFrame],
                 val_data: Union[str, pd.DataFrame],
                 tokens: Union[str, pd.DataFrame],
                 text_column: str,
                 seq_len: int,
                 test_data_feather_path: Optional[str] = None,
                 batch_size: Optional[int] = 1,
                 num_workers: Optional[int] = 1,
                 pin_memory: Optional[bool] = False,
                 persistent_workers: Optional[bool] = False,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = True) -> None:
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokens = tokens
        self.text_column = text_column
        self.seq_len = seq_len
        self.test_data_feather_path = test_data_feather_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.sample_size = sample_size
        self.check_normalization = check_normalization

    def setup(self,
              stage: Optional[str] = 'fit') -> None:
        if stage == 'fit' or stage is None:
            self._train_dataset = OneHotCharDataset(self.train_data,
                                                    self.tokens,
                                                    self.text_column,
                                                    self.seq_len,
                                                    sample_size=self.sample_size,
                                                    check_normalization=self.check_normalization)
            self._val_dataset = OneHotCharDataset(self.val_data,
                                                  self.tokens,
                                                  self.text_column,
                                                  self.seq_len,
                                                  sample_size=self.sample_size,
                                                  check_normalization=self.check_normalization)

        if stage == 'test':
            raise NotImplementedError('Should not be using test dataset yet.')

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self._train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=self.persistent_workers)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # Best pracice for val/test sets is to have shuffle=False.
        val_loader = DataLoader(self._val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                persistent_workers=self.persistent_workers)
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError('Should not be using test dataset yet.')


class EmbeddingDataModule(pl.LightningDataModule):
    """Simple pl.LightningDataModule wrapper for torch embedded text datasets.

    Args
    ----------
    `train_data`: pd.DataFrame or str
        DataFrame object or path to training data feather file.
    `val_data`: pd.DataFrame or str
        DataFrame object or path to training data feather file.
    `tokens`: pd.DataFrame or str
        DataFrame object or path to vocabulary feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        How many characters are read in from the text.
    `batch_size`: int, default = 1
        Dataloader batch size.
    `num_workers`: int, default = 1
        Dataloader's number of workers used.
    `pin_memory`: bool, default = False
        Dataloader's pin_memory option.
    `persistent_workers`: bool, default = False
        Dataloader's persistent_workers option.
    `sample_size`: int or float or None, optional, default = None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = True
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods
    ----------
    `test_dataloader`
        Not-yet implemented as to protect against accidentally viewing test set.
     """

    def __init__(self,
                 train_data: Union[str, pd.DataFrame],
                 val_data: Union[str, pd.DataFrame],
                 tokens: Union[str, pd.DataFrame],
                 text_column: str,
                 seq_len: int,
                 test_data_feather_path: Optional[str] = None,
                 batch_size: Optional[int] = 1,
                 num_workers: Optional[int] = 1,
                 pin_memory: Optional[bool] = False,
                 persistent_workers: Optional[bool] = False,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = True) -> None:
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokens = tokens
        self.text_column = text_column
        self.seq_len = seq_len
        self.test_data_feather_path = test_data_feather_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.sample_size = sample_size
        self.check_normalization = check_normalization

    def setup(self,
              stage: Optional[str] = 'fit') -> None:
        if stage == 'fit' or stage is None:
            self._train_dataset = EmbeddingDataset(self.train_data,
                                                   self.tokens,
                                                   self.text_column,
                                                   self.seq_len,
                                                   sample_size=self.sample_size,
                                                   check_normalization=self.check_normalization)
            self._val_dataset = EmbeddingDataset(self.val_data,
                                                 self.tokens,
                                                 self.text_column,
                                                 self.seq_len,
                                                 sample_size=self.sample_size,
                                                 check_normalization=self.check_normalization)

        if stage == 'test':
            raise NotImplementedError('Should not be using test dataset yet.')

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self._train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=self.persistent_workers)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # Best pracice for val/test sets is to have shuffle=False.
        val_loader = DataLoader(self._val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                persistent_workers=self.persistent_workers)
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError('Should not be using test dataset yet.')

# BELOW: Datamodules for testing architectures' ability to cheat. Not exported with *.


class EmbeddingDataCheatingModule(EmbeddingDataModule):
    """For use when testing the ability of a model to cheat via technical clues only.
    """

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        kwargs = {**kwargs, **{'check_normalization': False}}
        super().__init__(*args, **kwargs)

    def setup(self,
              stage: Optional[str] = 'fit') -> None:
        if stage == 'fit' or stage is None:
            self._train_dataset = EmbeddingCheatingDataset(self.train_data,
                                                           self.tokens,
                                                           self.text_column,
                                                           self.seq_len,
                                                           sample_size=self.sample_size,
                                                           check_normalization=self.check_normalization)
            self._val_dataset = EmbeddingCheatingDataset(self.val_data,
                                                         self.tokens,
                                                         self.text_column,
                                                         self.seq_len,
                                                         sample_size=self.sample_size,
                                                         check_normalization=self.check_normalization)

        if stage == 'test':
            raise NotImplementedError('Should not be using test dataset yet.')


class OneHotCharDataCheatingModule(OneHotCharDataModule):
    """For use when testing the ability of a model to cheat via technical clues
    only.
    """

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        kwargs = {**kwargs, **{'check_normalization': False}}
        super().__init__(*args, **kwargs)

    def setup(self,
              stage: Optional[str] = 'fit') -> None:
        if stage == 'fit' or stage is None:
            self._train_dataset = OneHotCharCheatingDataset(self.train_data,
                                                            self.tokens,
                                                            self.text_column,
                                                            self.seq_len,
                                                            sample_size=self.sample_size,
                                                            check_normalization=self.check_normalization)
            self._val_dataset = OneHotCharCheatingDataset(self.val_data,
                                                          self.tokens,
                                                          self.text_column,
                                                          self.seq_len,
                                                          sample_size=self.sample_size,
                                                          check_normalization=self.check_normalization)

        if stage == 'test':
            raise NotImplementedError('Should not be using test dataset yet.')
