from typing import Dict, Optional

from pandas import DataFrame
import torch
from torch import Tensor

from .text_normalizer import text_normalizer

BLANK_IDX = 0


def str_to_one_hot(
    s: str,
    char_to_idx: Dict[str, int],
    seq_len: Optional[int] = None,
    check_normalization: bool = False,
    strip_before_normalization_check: bool = False,
) -> Tensor:
    """One-hot-encode a string `s`.

    Description
    ----------
    Given the char_to_idx dictionary mapping chars to ints, return the one-hot
    encoded text.  Optionally force text to be seq_len long. Spaces (used for
    padding) and unknown characters are both mapped to a vector pointing in the
    0-direction. Text is expected to have been passed through text_normalizer
    first.  Resulting shape is (seq_len, len(char_to_idx)).

    Parameters
    ----------
    s : str
        Text to to be embedded
    char_to_idx : dict
        Mapping from chars to indices.
    seq_len : int or None, default 'None'
        Force text to be of length seq_len, if not None.
    check_normalization : bool, default False
        Verify the text was normalized as expected.
    strip_before_normalization_check : bool, default False
        Flag for whether to strip text before performing normalization check.

    Returns
    ----------
    s_tensor : Tensor
        One-hot encoded tensor.
    """
    if check_normalization:
        s_check = s.strip() if strip_before_normalization_check else s
        assert s_check == text_normalizer(
            s_check
        ), "String not normalized as expected, apply text_normalizer first."

    input_size = len(char_to_idx)
    s_len = len(s)
    if seq_len is None:
        seq_len = s_len
    else:
        if s_len > seq_len:
            s = s[:seq_len]
        else:
            s = (seq_len - s_len) * " " + s
    s_int = [char_to_idx.get(c, BLANK_IDX) for c in s]
    s_tensor = torch.zeros((seq_len, input_size))
    s_tensor[torch.arange(seq_len), s_int] = 1.0
    return s_tensor


def one_hot_to_str(tensor: Tensor, idx_to_char: Dict[int, str]) -> str:
    """Decode a one-hot-encoded tensor.

    Description
    ----------
    Given the idx_to_char dictionary mapping ints to chars, return the string
    corresponding to the one-hot encoded tensor. tensor is expected to be of
    shape (seq_len, num_chars)

    Parameters
    ----------
    s : str
        Text to to be embedded
    idx_to_char : dict
        Mapping from chars to indices.

    Returns
    ----------
    text : str
        Decoded string.
    """
    text = "".join(idx_to_char.get(c.argmax().item(), "<UNK>") for c in tensor)
    return text


def char_to_idx_from_df(
    df: DataFrame,
    char_col: str = "char",
    idx_col: str = "idx",
    check_normalization=True,
) -> dict:
    """Generates char_to_idx dictionary from DataFrame.

    Parameters
    ----------
    df : DataFrame
        Dataframe holding char-idx mapping
    char_col : str, optional
        Column in df holding characters.
    idx_col : str, optional
        Column in df holding indices.
    check_normalization : bool, default True
        Verify that blank space maps to zero.

    Returns
    -------
    dict
        Dictionary with (char, idx) as (key, value) pairs.
    """
    char_to_idx = {}
    for _, row in df.iterrows():
        char_to_idx[row[char_col]] = row[idx_col]
    if check_normalization:
        blank_check = char_to_idx[" "]
        assert (
            blank_check == BLANK_IDX
        ), f"Blank space expected to map to {BLANK_IDX}, not {blank_check}."
    return char_to_idx


def idx_to_char_from_df(
    df: DataFrame,
    char_col: str = "char",
    idx_col: str = "idx",
    check_normalization=True,
) -> dict:
    """Generates idx_to_char dictionary from DataFrame.

    Parameters
    ----------
    df : DataFrame
        Dataframe holding char-idx mapping
    char_col : str, optional
        Column in df holding characters.
    idx_col : str, optional
        Column in df holding indices.
    check_normalization : bool, default True
        Verify that zero space maps to blank space.

    Returns
    -------
    dict
        Dictionary with (idx, char) as (key, value) pairs.
    """
    idx_to_char = {}
    for _, row in df.iterrows():
        idx_to_char[row[idx_col]] = row[char_col]
    if check_normalization:
        blank_check = idx_to_char[0]
        assert (
            blank_check == BLANK_IDX
        ), f"0 expected to map to blank space, not {blank_check}."
    return idx_to_char
