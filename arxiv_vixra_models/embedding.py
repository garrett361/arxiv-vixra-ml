import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Union

from .text_normalizer import text_normalizer


def string_to_ints(s: str,
                   word_to_idx: Union[str, pd.DataFrame, Dict[str, int]],
                   seq_len: Optional[Union[None, int]] = None,
                   check_normalization: Optional[bool] = True) -> torch.Tensor:
    """Encode a string `s`, given the dict mapping words to ints or the
    DataFrame or path to the feather file containing the DataFrame with `'word'`
    and `'idx'` columns. Optionally force to be seq_len long. Padding by spaces
    and unknown words are both mapped to a vector pointing in the
    0-direction. Text is expected to have been passed through text_normalizer
    first.
    """
    # Test whether text was properly normalized.
    if check_normalization:
        assert s == text_normalizer(
            s), 'String not normalized as expected, apply text_normalizer first.'

    if not isinstance(word_to_idx, dict):
        if isinstance(word_to_idx, str):
            word_df = pd.read_feather(word_to_idx)
        else:
            word_df = word_to_idx
        word_to_idx = dict(
            zip(word_df['word'], np.arange(2, len(word_df) + 2)))

    # input_size is equal to mapping length plus 2, since 0 is reserved for
    # padding and 1 for unknown words.
    input_size = len(word_to_idx) + 2
    # Forcing sequence to be seq_len words long, if seq_len is not None.
    s_list = s.split()
    s_len = len(s_list)
    if seq_len is None:
        seq_len = s_len
    # Convert to integers. # 1 reserved for unknown words.
    if seq_len > s_len:
        s_int_list = (seq_len - s_len) * [0] + \
            [word_to_idx.get(w, 1) for w in s_list]
    else:
        s_int_list = [word_to_idx.get(w, 1) for w in s_list[:seq_len]]
    s_int_tensor = torch.tensor(s_int_list)
    return s_int_tensor


def ints_to_string(tensor: torch.Tensor,
                   idx_to_word: Union[str, pd.DataFrame, Dict[int, str]]) -> str:
    """Decode a integer-entry `tensor` to a `str` given the dict mapping ints to chars,
    the DataFrame or path to the feather file containing the DataFrame with 
    `'char'` and `'idx'` columns.  Assumes padding and unknown characters are 
    both mapped to a vector pointing in the 0-direction.
    """

    if not isinstance(idx_to_word, dict):
        if isinstance(idx_to_word, str):
            word_df = pd.read_feather(idx_to_word)
        else:
            word_df = idx_to_word
        idx_to_word = dict(
            zip(np.arange(2, len(word_df) + 2), word_df['word']))

    def get_string(num: int) -> str:
        if num == 0:
            return ''
        if num == 1:
            return '<UNK>'
        return idx_to_word[num]

    stripped_text = ' '.join(get_string(n.item()) for n in tensor).strip()
    return stripped_text
