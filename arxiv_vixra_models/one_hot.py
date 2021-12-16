import pandas as pd
import torch
from typing import Dict, Optional, Union

from .text_normalizer import text_normalizer


def one_hot_encoding(s: str,
                     char_to_idx: Union[str, pd.DataFrame, Dict[str, int]],
                     seq_len: Optional[Union[None, int]] = None,
                     check_normalization: Optional[bool] = True) -> torch.Tensor:
    """One-hot-encode a string `s`, given the dict mapping chars to ints or the
    DataFrame or path to the feather file containing the DataFrame with `'char'`
    and `'idx'` columns. Optionally force to be seq_len long. Padding by spaces
    and unknown characters are both mapped to a vector pointing in the
    0-direction. Text is expected to have been passed through text_normalizer
    first.
    """
    # Test whether text was properly normalized.
    if check_normalization:
        assert s == text_normalizer(
            s), 'String not normalized as expected, apply text_normalizer first.'

    if not isinstance(char_to_idx, dict):
        if isinstance(char_to_idx, str):
            char_df = pd.read_feather(char_to_idx)
        else:
            char_df = char_to_idx
        char_to_idx = dict(zip(char_df['char'], char_df['idx']))

    # Assuming that ' ' is a key mapping to 0 in char_to_idx
    assert ' ' in char_to_idx, "' ' should be a key in char_to_idx."
    assert char_to_idx[' '] == 0, f"' ' should be mapped to 0, not {char_to_idx[' ']} in char_to_idx."
    # input_size is equal to mapping length with the above assumption.
    input_size = len(char_to_idx)

    # Fitting to seq_len, if not None
    s_len = len(s)
    if seq_len is None:
        seq_len = s_len
    else:
        if s_len > seq_len:
            s = s[:seq_len]
        else:
            s = (seq_len - s_len) * ' ' + s
    # Convert to integers
    # All char_to_idx should be in char_to_idx, so don't use .get() with a default.
    # Small sanity check.
    s_int = [char_to_idx[c] for c in s]
    # Convert to a (seq_len, vocab_size) tensor which is one-hot encoded.
    s_tensor = torch.zeros((seq_len, input_size))
    s_tensor[torch.arange(seq_len), s_int] = 1.
    return s_tensor


def one_hot_decoding(tensor: torch.Tensor,
                     idx_to_char: Union[str, pd.DataFrame, Dict[int, str]]) -> str:
    """Decode one-hot `tensor` to a `str` given the dict mapping ints to chars,
    the DataFrame or path to the feather file containing the DataFrame with 
    `'char'` and `'idx'` columns.  Assumes padding and unknown characters are 
    both mapped to a vector pointing in the 0-direction.
    """

    if not isinstance(idx_to_char, dict):
        if isinstance(idx_to_char, str):
            char_df = pd.read_feather(idx_to_char)
        else:
            char_df = idx_to_char
        idx_to_char = dict(zip(char_df['idx'], char_df['char']))
    # We assume that 0 is mapped to ' '
    assert 0 in idx_to_char, "0 should be a key in idx_to_char"
    assert idx_to_char[0] == ' ', f"0 should map to ' ', not {idx_to_char[0]}, in idx_to_char."
    # All idxs should be in idx_to_char, so don't use .get() with a default.
    # Don't strip here, in order to sanity check model inputs for errant white space.
    text = ''.join(idx_to_char[c.argmax().item()] for c in tensor)
    return text
