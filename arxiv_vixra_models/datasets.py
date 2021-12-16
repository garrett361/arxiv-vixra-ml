import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union

from .one_hot import one_hot_encoding
from .embedding import string_to_ints


class OneHotCharDataset(Dataset):
    """Dataset subclass for processing text in feather/csv files at character level.

    Description
    ----------
    Assumes each row in the data feather file has 'source' column which
    is either 'arxiv' or 'vixra' and that a separate feather file mapping characters
    to integers (1-indexing assumed) exists. Outputs one-hot-encoded characters.

    Args
    ----------
    `data`: str or pd.DataFrame
        DataFrame object or path to data feather file.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to character feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        How many characters are read in from the text.
    `sample_size`: int or float or None, optional, default=None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = True
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods
    ----------
    `__getitem__`
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        one-hot encoded text and `source_int` is False, True if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
     """

    def __init__(self,
                 data: Union[str, pd.DataFrame],
                 tokens: Union[str, pd.DataFrame],
                 text_column: str,
                 seq_len: int,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = True) -> None:
        super().__init__()
        if isinstance(data, str):
            self.data = pd.read_feather(data, columns=[text_column, 'source'])
        else:
            # Important to copy, otherwise data input is modified.
            self.data = data.copy()
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        self._char_to_idx = dict(
            zip(self.tokens_df['char'], self.tokens_df['idx']))
        self.text_column = text_column
        self.seq_len = seq_len
        if sample_size is not None:
            # .copy() needed in below steps to avoid onnx export errors.
            if isinstance(sample_size, float):
                self.data = self.data.sample(frac=sample_size).copy()
            if isinstance(sample_size, int):
                self.data = self.data.sample(sample_size).copy()
        self.check_normalization = check_normalization

        # Perform one-hot encoding and encoding the source for arxiv/vixra.
        self.data['source'] = self.data['source'].apply(
            self._arxiv_vixra_encoding)
        self.data[text_column] = self.data[text_column].apply(
            self._one_hot_encoding)

    def _arxiv_vixra_encoding(self,
                              s: str) -> torch.Tensor:
        if s == 'vixra':
            return torch.tensor(True)
        if s == 'arxiv':
            return torch.tensor(False)
        raise ValueError(
            'Source string must either be arxiv or vixra, invalid data.')

    def _one_hot_encoding(self,
                          s: str) -> torch.Tensor:
        return one_hot_encoding(s, self._char_to_idx, self.seq_len, self.check_normalization)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.data.iloc[idx].loc['source']
        one_hot_text = self.data.iloc[idx].loc[self.text_column]

        return one_hot_text, source


class EmbeddingDataset(Dataset):
    """Dataset subclass for processing text in feather/csv files at word level.

    Description
    ----------

    Assumes each row in the data feather file has 'source' column which
    is either 'arxiv' or 'vixra' and that a separate feather file mapping words
    to integers (1-indexing assumed) exists. Outputs integer-encoded words.

    Args
    ----------
    `data`: str or pd.DataFrame
        DataFrame object or path to data feather file.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to vocabulary feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        Maximum number of words included from the text in a sequence.
    `sample_size`: int or float or None, optional, default=None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = True
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods 
    ----------
    `__getitem__`
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        integer-encoded text and `source_int` is False, True if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
     """

    def __init__(self,
                 data: Union[str, pd.DataFrame],
                 tokens: Union[str, pd.DataFrame],
                 text_column: str,
                 seq_len: int,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = True) -> None:
        super().__init__()
        if isinstance(data, str):
            self.data = pd.read_feather(data, columns=[text_column, 'source'])
        else:
            # Important to copy, otherwise data input is modified.
            self.data = data.copy()
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        # Create dictionary from tokens. Start from 2, accounting for padding and
        # unknown maps at 0 and 1.
        self._word_to_idx = dict(
            zip(self.tokens_df['word'], np.arange(2, len(self.tokens_df) + 2)))

        self.text_column = text_column
        self.seq_len = seq_len
        if sample_size is not None:
            # .copy() needed in below steps to avoid onnx export errors.
            if isinstance(sample_size, float):
                self.data = self.data.sample(frac=sample_size).copy()
            if isinstance(sample_size, int):
                self.data = self.data.sample(sample_size).copy()
        self.check_normalization = check_normalization

        # Perform embedding and encoding the source for arxiv/vixra.
        self.data['source'] = self.data['source'].apply(
            self._arxiv_vixra_encoding)
        self.data[text_column] = self.data[text_column].apply(
            self._string_to_ints)

    def _arxiv_vixra_encoding(self,
                              s: str) -> torch.Tensor:
        if s == 'vixra':
            return torch.tensor(True)
        if s == 'arxiv':
            return torch.tensor(False)
        raise ValueError(
            'Source string must either be arxiv or vixra, invalid data.')

    def _string_to_ints(self,
                        s: str) -> torch.Tensor:
        return string_to_ints(s,
                              self._word_to_idx,
                              self.seq_len,
                              self.check_normalization)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.data.iloc[idx].loc['source']
        tokenized_text = self.data.iloc[idx].loc[self.text_column]

        return tokenized_text, source

# Datasets below are for testing the ability of models to cheat.


class OneHotCharCheatingDataset(Dataset):
    """Dataset subclass for processing text in feather/csv files at character level
    and testing the model's ability to cheat via technical clues by inserting
    a blank space at the end of all vixra samples.

    Description
    ----------

    Assumes each row in the data feather file has 'source' column which
    is either 'arxiv' or 'vixra' and that a separate feather file mapping characters
    to integers (1-indexing assumed) exists. Outputs one-hot-encoded characters.

    Args
    ----------
    `data`: str or pd.DataFrame
        DataFrame object or path to data feather file.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to character feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        How many characters are read in from the text.
    `sample_size`: int or float or None, optional, default=None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = False
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods 
    ----------
    `__getitem__`
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        one-hot encoded text and `source_int` is False, True if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
     """

    def __init__(self,
                 data: Union[str, pd.DataFrame],
                 tokens: Union[str, pd.DataFrame],
                 text_column: str,
                 seq_len: int,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = False) -> None:
        super().__init__()
        if isinstance(data, str):
            self.data = pd.read_feather(data, columns=[text_column, 'source'])
        else:
            # Important to copy, otherwise data input is modified.
            self.data = data.copy()
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        self._char_to_idx = dict(
            zip(self.tokens_df['char'], self.tokens_df['idx']))
        self.text_column = text_column
        self.seq_len = seq_len
        if sample_size is not None:
            # .copy() needed in below steps to avoid onnx export errors.
            if isinstance(sample_size, float):
                self.data = self.data.sample(frac=sample_size).copy()
            if isinstance(sample_size, int):
                self.data = self.data.sample(sample_size).copy()
        self.check_normalization = check_normalization

        # Perform one-hot encoding and encoding the source for arxiv/vixra.
        self.data['source'] = self.data['source'].apply(
            self._arxiv_vixra_encoding)
        self.data[text_column] = self.data[text_column].apply(
            self._one_hot_encoding)

        # ** INSERTING CHEAT HERE **
        def insert_trailing_space(t):
            t = t.clone()
            t[-1] = torch.zeros_like(t[-1])
            t[-1, 0] = 1
            return t
        # pd fails to a zero-dim broadcast torch.tensor(True) on the RHS
        vixra_mask = (self.data['source'].values == [
                      self._arxiv_vixra_encoding('vixra') for _ in self.data['source']])
        self.data.loc[vixra_mask, text_column] = self.data.loc[vixra_mask,
                                                               text_column].apply(insert_trailing_space)

    def _arxiv_vixra_encoding(self,
                              s: str) -> torch.Tensor:
        if s == 'vixra':
            return torch.tensor(True)
        if s == 'arxiv':
            return torch.tensor(False)
        raise ValueError(
            'Source string must either be arxiv or vixra, invalid data.')

    def _one_hot_encoding(self,
                          s: str) -> torch.Tensor:
        return one_hot_encoding(s,
                                self._char_to_idx,
                                self.seq_len,
                                self.check_normalization)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.data.iloc[idx].loc['source']
        one_hot_text = self.data.iloc[idx].loc[self.text_column]

        return one_hot_text, source


class EmbeddingCheatingDataset(Dataset):
    """Dataset subclass for processing text in feather/csv files at word level
    and testing the model's ability to cheat via technical clues by inserting
    a blank space at the end of all vixra samples.

    Description
    ----------

    Assumes each row in the data feather file has 'source' column which
    is either 'arxiv' or 'vixra' and that a separate feather file mapping words
    to integers (1-indexing assumed) exists. Outputs integer-encoded words.

    Args
    ----------
    `data`: str or pd.DataFrame
        DataFrame object or path to data feather file.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to vocabulary feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        Maximum number of words included from the text in a sequence.
    `sample_size`: int or float or None, optional, default=None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = False
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods 
    ----------
    `__getitem__`
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        integer-encoded text and `source_int` is False, True if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
     """

    def __init__(self,
                 data: Union[str, pd.DataFrame],
                 tokens: Union[str, pd.DataFrame],
                 text_column: str,
                 seq_len: int,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = False) -> None:
        super().__init__()
        if isinstance(data, str):
            self.data = pd.read_feather(data, columns=[text_column, 'source'])
        else:
            # Important to copy, otherwise data input is modified.
            self.data = data.copy()
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        # Create dictionary from tokens. Start from 2, accounting for padding and
        # unknown maps at 0 and 1.
        self._word_to_idx = dict(
            zip(self.tokens_df['word'], np.arange(2, len(self.tokens_df) + 2)))

        self.text_column = text_column
        self.seq_len = seq_len
        if sample_size is not None:
            # .copy() needed in below steps to avoid onnx export errors.
            if isinstance(sample_size, float):
                self.data = self.data.sample(frac=sample_size).copy()
            if isinstance(sample_size, int):
                self.data = self.data.sample(sample_size).copy()
        self.check_normalization = check_normalization

        # Perform embedding and encoding the source for arxiv/vixra.
        self.data['source'] = self.data['source'].apply(
            self._arxiv_vixra_encoding)
        self.data[text_column] = self.data[text_column].apply(
            self._string_to_ints)

        # ** INSERTING CHEAT HERE **
        def insert_trailing_space(t):
            t = t.clone()
            t[-1] = 0
            return t
        # pd fails to a zero-dim broadcast torch.tensor(True) on the RHS
        vixra_mask = (self.data['source'].values == [
                      self._arxiv_vixra_encoding('vixra') for _ in self.data['source']])
        self.data.loc[vixra_mask, text_column] = self.data.loc[vixra_mask,
                                                               text_column].apply(insert_trailing_space)

    def _arxiv_vixra_encoding(self,
                              s: str) -> torch.Tensor:
        if s == 'vixra':
            return torch.tensor(True)
        if s == 'arxiv':
            return torch.tensor(False)
        raise ValueError(
            'Source string must either be arxiv or vixra, invalid data.')

    def _string_to_ints(self,
                        s: str) -> torch.Tensor:
        return string_to_ints(s,
                              self._word_to_idx,
                              self.seq_len,
                              self.check_normalization)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.data.iloc[idx].loc['source']
        tokenized_text = self.data.iloc[idx].loc[self.text_column]

        return tokenized_text, source
