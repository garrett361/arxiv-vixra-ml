from typing import Optional, Union

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import (
    EmbeddingDatasetAV,
    EmbeddingDatasetNextLM,
    OneHotCharDatasetAV,
    OneHotCharDatasetNextLM,
)


class OneHotCharDataModuleAV(pl.LightningDataModule):
    """DataModule for arxiv/vixra classification using
    one-hot encoded, character-level text.

    Description
    ----------
    Data and the 1-indexed char dict are expected to be in feather format.
    The `setup` method initializes the appropriate Dataset objects, which are
    assumed to be returning one-hot encoded text via their `__getitem__`
    methods.

    Parameters
    ----------
    train_data : pd.DataFrame or str
        DataFrame object or path to training data feather file.
    val_data :  pd.DataFrame or str
        DataFrame object or path to training data feather file.
    tokens : pd.DataFrame or str
        DataFrame object or path to character feather file.
    text_column : str
        Column in data feather file containing text, e.g. `'title'`.
    seq_len : int
        Sequence length used for processing text.
    batch_size : int, default 1
        Dataloader batch size.
    num_workers : int, default 1
        Dataloader's number of workers used.
    pin_memory : bool, default False
        Dataloader's pin_memory option.
    persistent_workers : bool, default False
        Dataloader's persistent_workers option.
    sample_size : int or float or None, default None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.

    Notes
    ----------
    test_dataloader
        Not-yet implemented method as to protect against accidentally viewing
        test set.
    """

    def __init__(
        self,
        train_data: Union[str, pd.DataFrame],
        val_data: Union[str, pd.DataFrame],
        tokens: Union[str, pd.DataFrame],
        text_column: str,
        seq_len: int,
        test_data_feather_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sample_size: Optional[Union[int, float]] = None,
        check_normalization: bool = True,
    ) -> None:
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

    def setup(self, stage: Optional[str] = "fit") -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = OneHotCharDatasetAV(
                data=self.train_data,
                tokens=self.tokens,
                text_column=self.text_column,
                seq_len=self.seq_len,
                sample_size=self.sample_size,
                check_normalization=self.check_normalization,
            )
            self._val_dataset = OneHotCharDatasetAV(
                data=self.val_data,
                tokens=self.tokens,
                text_column=self.text_column,
                seq_len=self.seq_len,
                sample_size=self.sample_size,
                check_normalization=self.check_normalization,
            )

        if stage == "test":
            raise NotImplementedError("Should not be using test dataset yet.")

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # Best pracice for val/test sets is to have shuffle=False.
        val_loader = DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Should not be using test dataset yet.")


class OneHotCharDataModuleNextLM(pl.LightningDataModule):
    """DataModule for handling one-hot-encoded character-level text for
    language-model generation via next-character prediction.


    Description
    ----------
    1-indexed char dict are expected to be in feather format.
    The `setup` method initializes the appropriate Dataset objects, which are
    assumed to be returning one-hot encoded text via their `__getitem__`
    methods.

    Parameters
    ----------
    train_text : str
        Text for training.
    val_text : str
        Text for validation.
    tokens : pd.DataFrame or str
        DataFrame object or path to character feather file.
    seq_len : int
        Sequence length used for processing text.
    batch_size : int, default 1
        Dataloader batch size.
    num_workers : int, default 1
        Dataloader's number of workers used.
    pin_memory : bool, default False
        Dataloader's pin_memory option.
    persistent_workers : bool, default False
        Dataloader's persistent_workers option.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.
    strip_before_normalization_check : bool, default True
        Flag for whether to strip text before performing normalization check.

    Notes
    ----------
    test_dataloader
        Not-yet implemented method as to protect against accidentally viewing
        test set.
    """

    def __init__(
        self,
        train_text: str,
        val_text: str,
        tokens: Union[str, pd.DataFrame],
        seq_len: int,
        test_data_feather_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        check_normalization: bool = True,
        strip_before_normalization_check: bool = True,
    ) -> None:
        super().__init__()
        self.train_text = train_text
        self.val_text = val_text
        self.tokens = tokens
        self.seq_len = seq_len
        self.test_data_feather_path = test_data_feather_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.check_normalization = check_normalization
        self.strip_before_normalization_check = strip_before_normalization_check

    def setup(self, stage: Optional[str] = "fit") -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = OneHotCharDatasetNextLM(
                text=self.train_text,
                tokens=self.tokens,
                seq_len=self.seq_len,
                check_normalization=self.check_normalization,
                strip_before_normalization_check=self.strip_before_normalization_check,
            )
            self._val_dataset = OneHotCharDatasetNextLM(
                text=self.val_text,
                tokens=self.tokens,
                seq_len=self.seq_len,
                check_normalization=self.check_normalization,
                strip_before_normalization_check=self.strip_before_normalization_check,
            )

        if stage == "test":
            raise NotImplementedError("Should not be using test dataset yet.")

    def train_dataloader(self) -> DataLoader:
        # Setting drop_last = True is important for avoiding end-of-epoch
        # loss blow-ups.
        train_loader = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # Best pracice for val/test sets is to have shuffle=False.
        val_loader = DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Should not be using test dataset yet.")


class EmbeddingDataModuleAV(pl.LightningDataModule):
    """DataModule for arxiv/vixra classification using word-level
    tokenized text.

    Description
    ----------
    tokens DataFrame (or associated feather file) is expected to have a 'count'
    column tallying the number of times each word appeared in the training set.

    Parameters
    ----------
    train_data : pd.DataFrame or str
        DataFrame object or path to training data feather file.
    val_data :  pd.DataFrame or str
        DataFrame object or path to training data feather file.
    tokens : pd.DataFrame or str
        DataFrame object or path to vocabulary feather file.
    min_word_count : int, default 1
        Minimum count for a word in tokens to be included in the vocabulary.
    text_column : str
        Column in data feather file containing text, e.g. `'title'`.
    seq_len : int
        Sequence length used for processing text.
    batch_size : int, default 1
        Dataloader batch size.
    num_workers : int, default 1
        Dataloader's number of workers used.
    pin_memory : bool, default False
        Dataloader's pin_memory option.
    persistent_workers : bool, default False
        Dataloader's persistent_workers option.
    sample_size : int or float or None, default None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.

    Notes
    ----------
    test_dataloader
        Not-yet implemented method as to protect against accidentally viewing
        test set.
    """

    def __init__(
        self,
        train_data: Union[str, pd.DataFrame],
        val_data: Union[str, pd.DataFrame],
        tokens: Union[str, pd.DataFrame],
        text_column: str,
        seq_len: int,
        min_word_count: int = 1,
        test_data_feather_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sample_size: Optional[Union[int, float]] = None,
        check_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokens = tokens
        self.min_word_count = min_word_count
        self.text_column = text_column
        self.seq_len = seq_len
        self.test_data_feather_path = test_data_feather_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.sample_size = sample_size
        self.check_normalization = check_normalization

    def setup(self, stage: Optional[str] = "fit") -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = EmbeddingDatasetAV(
                data=self.train_data,
                tokens=self.tokens,
                min_word_count=self.min_word_count,
                text_column=self.text_column,
                seq_len=self.seq_len,
                sample_size=self.sample_size,
                check_normalization=self.check_normalization,
            )
            self._val_dataset = EmbeddingDatasetAV(
                data=self.val_data,
                tokens=self.tokens,
                min_word_count=self.min_word_count,
                text_column=self.text_column,
                seq_len=self.seq_len,
                sample_size=self.sample_size,
                check_normalization=self.check_normalization,
            )

        if stage == "test":
            raise NotImplementedError("Should not be using test dataset yet.")

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # Best pracice for val/test sets is to have shuffle=False.
        val_loader = DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Should not be using test dataset yet.")


class EmbeddingDataModuleNextLM(pl.LightningDataModule):
    """DataModule for handling word-level tokenized text for
    language-model generation via next-word prediction.

    Description
    ----------
    tokens DataFrame (or associated feather file) is expected to have a 'count'
    column tallying the number of times each word appeared in the training set.

    Parameters
    ----------
    train_text : str
        Text for training.
    val_text : str
        Text for validation.
    tokens : pd.DataFrame or str
        DataFrame object or path to vocabulary feather file.
    min_word_count : int, default 1
        Minimum count for a word in tokens to be included in the vocabulary.
    seq_len : int
        Sequence length used for processing text.
    batch_size : int, default 1
        Dataloader batch size.
    num_workers : int, default 1
        Dataloader's number of workers used.
    pin_memory : bool, default False
        Dataloader's pin_memory option.
    persistent_workers : bool, default False
        Dataloader's persistent_workers option.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.
    strip_before_normalization_check : bool, default True
        Flag for whether to strip text before performing normalization check.

    Notes
    ----------
    test_dataloader
        Not-yet implemented method as to protect against accidentally viewing
        test set.
    """

    def __init__(
        self,
        train_text: str,
        val_text: str,
        tokens: Union[str, pd.DataFrame],
        seq_len: int,
        min_word_count: int = 1,
        test_data_feather_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        check_normalization: bool = True,
        strip_before_normalization_check: bool = True,
    ) -> None:
        super().__init__()
        self.train_text = train_text
        self.val_text = val_text
        self.tokens = tokens
        self.min_word_count = min_word_count
        self.seq_len = seq_len
        self.test_data_feather_path = test_data_feather_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.check_normalization = check_normalization
        self.strip_before_normalization_check = strip_before_normalization_check

    def setup(self, stage: Optional[str] = "fit") -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = EmbeddingDatasetNextLM(
                text=self.train_text,
                tokens=self.tokens,
                min_word_count=self.min_word_count,
                seq_len=self.seq_len,
                check_normalization=self.check_normalization,
                strip_before_normalization_check=self.strip_before_normalization_check,
            )
            self._val_dataset = EmbeddingDatasetNextLM(
                text=self.val_text,
                tokens=self.tokens,
                min_word_count=self.min_word_count,
                seq_len=self.seq_len,
                check_normalization=self.check_normalization,
                strip_before_normalization_check=self.strip_before_normalization_check,
            )

        if stage == "test":
            raise NotImplementedError("Should not be using test dataset yet.")

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # Best pracice for val/test sets is to have shuffle=False.
        val_loader = DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Should not be using test dataset yet.")
