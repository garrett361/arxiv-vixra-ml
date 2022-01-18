from copy import deepcopy
from typing import Tuple, Optional, Union, Dict
import warnings

from pandas import DataFrame
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .embedding_utils import str_to_idxs, word_to_idx_dict_from_df
from .one_hot_utils import str_to_one_hot, char_to_idx_dict_from_df

PAD_IDX = 0
UNK_IDX = 1
BLANK_IDX = 0


class OneHotCharDatasetAV(Dataset):
    """Dataset for handling one-hot-encoded character-level text for
    arxiv/vixra classification.

    Description
    ----------
    Assumes each row in text_df has 'source' column which
    is either 'arxiv' or 'vixra'. Uses `tokens_df` to map characters to
    indices.

    Parameters
    ----------
    text_df : DataFrame
        DataFrame containing text.
    tokens_df : DataFrame
        Character-to-index data stored in 'char' and 'idx' columns.
    text_column : str
        Column text DataFrames containing  desired text, e.g. `'title'`.
    seq_len : int
        Sequence length used for processing text.
    sample_size : int or float, optional
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.

    Methods
    ----------
    __getitem__
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        one-hot encoded text and `source_int` is False, True if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
    """

    def __init__(
        self,
        text_df: DataFrame,
        tokens_df: DataFrame,
        text_column: str,
        seq_len: int,
        sample_size: Optional[Union[int, float]] = None,
        check_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.text_column = text_column
        self.seq_len = seq_len

        # Important to copy, otherwise data input is modified.
        self.text_df = deepcopy(text_df)
        self.tokens_df = deepcopy(tokens_df)
        self._char_to_idx_dict = char_to_idx_dict_from_df(self.tokens_df)

        if sample_size is not None:
            if isinstance(sample_size, float):
                self.text_df = self.text_df.sample(frac=sample_size)
            if isinstance(sample_size, int):
                self.text_df = self.text_df.sample(sample_size)
        self.check_normalization = check_normalization

        # Perform one-hot encoding and encoding the source for arxiv/vixra.
        self.text_df["source"] = self.text_df["source"].apply(
            self._arxiv_vixra_encoding
        )
        self.text_df[text_column] = self.text_df[text_column].apply(
            self._str_to_one_hot
        )

    def _arxiv_vixra_encoding(self, s: str) -> Tensor:
        if s == "vixra":
            return torch.tensor(True)
        if s == "arxiv":
            return torch.tensor(False)
        raise ValueError("Source string must either be arxiv or vixra, invalid data.")

    def _str_to_one_hot(self, s: str) -> Tensor:
        return str_to_one_hot(
            s, self._char_to_idx_dict, self.seq_len, self.check_normalization
        )

    def __len__(self) -> int:
        return len(self.text_df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        source = self.text_df.iloc[idx].loc["source"]
        one_hot_text = self.text_df.iloc[idx].loc[self.text_column]

        return one_hot_text, source


class OneHotCharDatasetNextLM(Dataset):
    """Dataset for handling one-hot-encoded character-level text for
    language-model generation via next-character prediction.

    Description
    ----------
    Uses `tokens` to map characters to integers.

    Parameters
    ----------
    text : str
        Training text.
    tokens_df : DataFrame
        Character-to-index data stored in 'char' and 'idx' columns.
    seq_len : int
        Sequence length used for processing text.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.
    strip_before_normalization_check : bool, default True
        Flag for whether to strip text before performing normalization check.

    Methods
    ----------
    __getitem__
        Returns `(text_tensor, next_text_classes)`  where `text_tensor` is the
        one-hot encoded text and `next_text_classes` is a tensor containing
        the correct class labels for the similar slice of text shifted one
        space to the future.
    """

    def __init__(
        self,
        text: str,
        tokens_df: DataFrame,
        seq_len: int,
        check_normalization: bool = True,
        strip_before_normalization_check: bool = True,
    ) -> None:
        super().__init__()
        self.text = text
        self.seq_len = seq_len
        self.tokens_df = deepcopy(tokens_df)
        self._char_to_idx_dict = char_to_idx_dict_from_df(self.tokens_df)
        self.check_normalization = check_normalization
        self.strip_before_normalization_check = strip_before_normalization_check

    def _str_to_one_hot(self, s: str) -> Tensor:
        return str_to_one_hot(
            s=s,
            char_to_idx_dict=self._char_to_idx_dict,
            seq_len=self.seq_len,
            check_normalization=self.check_normalization,
            strip_before_normalization_check=self.strip_before_normalization_check,
        )

    def _get_classes_tensor(self, s: str) -> Tensor:
        classes = [self._char_to_idx_dict[ch] for ch in s]
        classes_t = torch.tensor(classes)
        return classes_t

    def __len__(self) -> int:
        return len(self.text) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start, stop = idx * self.seq_len, (idx + 1) * self.seq_len
        text = self.text[start:stop]
        next_text = self.text[start + 1 : stop + 1]
        text_tensor = self._str_to_one_hot(text)
        next_text_classes = self._get_classes_tensor(next_text)

        return text_tensor, next_text_classes


class EmbeddingDatasetAV(Dataset):
    """Dataset for handling word-level tokenized text for arxiv/vixra
    classification.

    Description
    ----------

    Assumes each row in text_df has 'source' column which
    is either 'arxiv' or 'vixra'. Outputs integer-encoded words. tokens_df DataFrame
    is expected to have a 'count' column tallying
    the number of times each word appeared in the training set and words to be
    sorted in descending order by count. Padding and <UNK> are assumed *not* to
    be in tokens.

    Parameters
    ----------
    text_df : DataFrame
        DataFrame containing text.
    tokens_df : DataFrame
        Token counts data stored in 'word' and 'count' columns.
    min_word_count : int, default 1
        Minimum count for a word in tokens_df to be included in the vocabulary.
    text_column : str
        Column text DataFrames containing  desired text, e.g. `'title'`.
    seq_len : int
        Sequence length used for processing text.
    sample_size : int or float, optional
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.

    Methods
    ----------
    __getitem__
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        integer-encoded text and `source_int` is False, True if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
    """

    def __init__(
        self,
        text_df: DataFrame,
        tokens_df: DataFrame,
        text_column: str,
        seq_len: int,
        min_word_count: int = 1,
        sample_size: Optional[Union[int, float]] = None,
        check_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.text_column = text_column
        self.seq_len = seq_len
        self.min_word_count = min_word_count
        self.sample_size = sample_size
        self.check_normalization = check_normalization

        # Important to copy, otherwise data input is modified.
        self.text_df = deepcopy(text_df)
        self.tokens_df = deepcopy(tokens_df)
        if min_word_count > 1:
            if "count" in self.tokens_df:
                self.tokens_df = self.tokens_df[
                    self.tokens_df["count"] >= min_word_count
                ]
            else:
                warnings.warn(
                    "count column does not exist in tokens_df DataFrame, min_word_count arg ignored."
                )

        # Create dictionary from tokens. Start from 2, accounting for padding and
        # unknown maps at 0 and 1.
        self._word_to_idx_dict = word_to_idx_dict_from_df(self.tokens_df)

        if sample_size is not None:
            if isinstance(sample_size, float):
                self.text_df = self.text_df.sample(frac=sample_size)
            if isinstance(sample_size, int):
                self.text_df = self.text_df.sample(sample_size)

        # Perform embedding and encoding the source for arxiv/vixra.
        self.text_df["source"] = self.text_df["source"].apply(
            self._arxiv_vixra_encoding
        )
        self.text_df[text_column] = self.text_df[text_column].apply(self._str_to_idxs)

    def _arxiv_vixra_encoding(self, s: str) -> Tensor:
        if s == "vixra":
            return torch.tensor(True)
        if s == "arxiv":
            return torch.tensor(False)
        raise ValueError("Source string must either be arxiv or vixra, invalid data.")

    def _str_to_idxs(self, s: str) -> Tensor:
        return str_to_idxs(
            s, self._word_to_idx_dict, self.seq_len, self.check_normalization
        )

    def __len__(self) -> int:
        return len(self.text_df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        source = self.text_df.iloc[idx].loc["source"]
        tokenized_text = self.text_df.iloc[idx].loc[self.text_column]

        return tokenized_text, source


class EmbeddingDatasetNextLM(Dataset):
    """Dataset for handling word-level tokenized text for
    language-model generation via next-word prediction.

    Description
    ----------

    Assumes each row in text_df has 'source' column which is either 'arxiv'
    or 'vixra'. Outputs integer-encoded words. tokens_df DataFrame is expected
    to have a 'count' column tallying the number of times each word appeared in
    the training set and words to be     sorted in descending order by count.
    Padding and <UNK> are assumed *not* to be in tokens.

    Parameters
    ----------
    text : str
        Training text.
    tokens_df : DataFrame
        Token counts data stored in 'word' and 'count' columns.
    min_word_count : int, default 1
        Minimum count for a word in tokens_df to be included in the vocabulary.
    seq_len : int
        Sequence length used for processing text.
    check_normalization : bool, default True
        Check whether the text data was normalized according to text_normalizer.
    strip_before_normalization_check : bool, default True
        Flag for whether to strip text before performing normalization check.

    Methods
    ----------
    __getitem__
        Returns `(text_tensor, next_text_classes)`  where `text_tensor` is the
        tokenized text tensor and `next_text_classes` is a tensor containing
        the correct class labels for the similar slice of text shifted one
        space to the future.
    """

    def __init__(
        self,
        text: str,
        tokens_df: DataFrame,
        seq_len: int,
        min_word_count: int = 1,
        check_normalization: bool = True,
        strip_before_normalization_check: bool = True,
    ) -> None:
        super().__init__()
        self.text = text
        self.split_text = text.split()
        self.seq_len = seq_len
        self.min_word_count = min_word_count
        self.check_normalization = check_normalization
        self.strip_before_normalization_check = strip_before_normalization_check

        self.tokens_df = deepcopy(tokens_df)
        if min_word_count > 1:
            if "count" in self.tokens_df:
                self.tokens_df = self.tokens_df[
                    self.tokens_df["count"] >= min_word_count
                ]
            else:
                warnings.warn(
                    "count column does not exist in tokens_df DataFrame, min_word_count arg ignored."
                )

        # Create dictionary from tokens. Start from 2, accounting for
        # padding and unknown maps at 0 and 1.
        self._word_to_idx_dict = word_to_idx_dict_from_df(self.tokens_df)

    def _str_to_idxs(self, s: str) -> Tensor:
        return str_to_idxs(
            s=s,
            word_to_idx_dict=self._word_to_idx_dict,
            seq_len=self.seq_len,
            check_normalization=self.check_normalization,
            strip_before_normalization_check=self.strip_before_normalization_check,
        )

    def _get_classes_tensor(self, s: str) -> Tensor:
        classes = [self._word_to_idx_dict[w] for w in s.split()]
        classes_t = torch.tensor(classes)
        return classes_t

    def __len__(self) -> int:
        return len(self.split_text) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start, stop = idx * self.seq_len, (idx + 1) * self.seq_len
        text = " ".join(self.split_text[start:stop])
        next_text = " ".join(self.split_text[start + 1 : stop + 1])
        text_tensor = self._str_to_idxs(text)
        next_text_classes = self._get_classes_tensor(next_text)

        return text_tensor, next_text_classes


class GloVeDataset(Dataset):
    """Dataset subclass for the GloVe algorithm.

    Description
    ----------

    Returns tensors of row and column indices and the corresponding elements
    from the co-occurrence matrix.

    Parameters
    ----------
    co_matrix : Tensor
        Co-occurrence matrix, a sparse torch tensor.

    Methods
    ----------
    __getitem__
        Returns a tuple of tensors: row, col, co_matrix_element, where
        co_matrix_element = co_matrix[row, col].
    """

    def __init__(self, co_matrix: Tensor) -> None:
        super().__init__()
        # __getitem__ is an order-of-magnitude faster from a dense tensor than
        # a sparse one.
        self._co_matrix = co_matrix.to_dense()
        self._indices = co_matrix.coalesce().indices()

    def __len__(self) -> int:
        """Return the number of non-trivial entries in the sparse co_matrix."""
        return self._indices.shape[-1]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns a pair of indices and the corresponding co_matrix entry."""
        row, col = self._indices[:, idx]
        co_matrix_element = self._co_matrix[row, col]
        return row, col, co_matrix_element


class CoMatrixDataset(Dataset):
    """Dataset for efficiently generating co-occurrence matrices.

    Description
    ----------

    Encoding performed using a word_to_idx_dict dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively.
    Text is expected to have been passed through text_normalizer first.
    Text normalization and proper mapping via word_to_idx_dict can be verified
    through the check_normalization flag.

    Parameters
    ----------
    text : str
        Text to to be embedded
    word_to_idx_dict : dict
        Mapping from words to indices.
    context_window : int, default 2
        Width of the context window used on either side of the center word.
    include_center_in_context : bool, default False
        Experimental flag for including the center word in its own context.

    Methods
    ----------
    __getitem__
        Returns a tuple of tensors: row, col, co_matrix_element, where
        co_matrix_element = co_matrix[row, col].
    """

    def __init__(
        self,
        text: str,
        word_to_idx_dict: Dict[str, int],
        context_window: int = 2,
        include_center_in_context: bool = False,
    ) -> None:
        super().__init__()
        print("Encoding text...")
        # Slicing a list of words then generating the appropriate Tensors
        # is about twice as fast as generating a large Tensor first and
        # slicing and concatenating it down.
        self._encoded_text = [
            word_to_idx_dict.get(word, UNK_IDX) for word in text.strip().split()
        ]
        print("...done!")
        self.context_window = context_window
        self.include_center_in_context = include_center_in_context

    def __len__(self) -> int:
        """Return the number of non-trivial entries in the sparse co_matrix."""
        return len(self._encoded_text) - 2 * self.context_window

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Returns the word idx of the center word and a left-to-right
        ordered tensor of its surrounding context words.
        """
        pos = idx + self.context_window
        center_idx = torch.tensor(self._encoded_text[pos])
        left, right = pos - self.context_window, pos + self.context_window + 1
        if self.include_center_in_context:
            context = torch.tensor(self._encoded_text[left:right])
        else:
            left_context = self._encoded_text[left:pos]
            right_context = self._encoded_text[pos + 1 : right]
            context = torch.tensor(left_context + right_context)
        return center_idx, context
