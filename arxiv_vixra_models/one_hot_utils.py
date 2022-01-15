from typing import Dict, Optional

import torch

from .text_normalizer import text_normalizer

PAD_IDX = 0
UNK_IDX = 1
BLANK_IDX = 0


def str_to_one_hot(
    s: str,
    char_to_idx: Dict[str, int],
    seq_len: Optional[int] = None,
    check_normalization: bool = False,
    strip_before_normalization_check: bool = False,
) -> torch.Tensor:
    """One-hot-encode a string `s`.

    Description
    ----------
    Given the char_to_idx dictionary mapping chars to ints, return the one-hot
    encoded text.  Optionally force text to be seq_len long. Spaces (used for
    padding) and unknown characters are both mapped to a vector pointing in the
    0-direction. Text is expected to have been passed through text_normalizer
    first and char_to_idx should map a blank space to zero.  Resulting shape is
    (seq_len, len(char_to_idx)).

    Args
    ----------
    `s`: str
        Text to to be embedded
    `char_to_idx`: dict
        Mapping from chars to indices.
    `seq_len`: int or None, default = 'None'
        Force text to be of length seq_len, if not None.
    `check_normalization`: bool, default = False
        Verify the text and char_to_idx mapping are of the proper formats.
    `strip_before_normalization_check`: bool, default = False
        Flag for whether to strip text before performing normalization check.

    Returns
    ----------
    `s_tensor`: torch.Tensor
        One-hot encoded tensor.
    """
    if check_normalization:
        s_check = s.strip() if strip_before_normalization_check else s
        assert s_check == text_normalizer(
            s_check
        ), "String not normalized as expected, apply text_normalizer first."
        blank_check = char_to_idx.get(" ", None)
        assert (
            blank_check == BLANK_IDX
        ), "char_to_idx expected to map blank space to 0, not {blank_check}."

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


def one_hot_to_str(
    tensor: torch.Tensor, idx_to_char: Dict[int, str], check_normalization: bool = False
) -> str:
    """Decode a one-hot-encoded tensor.

    Description
    ----------
    Given the idx_to_char dictionary mapping ints to chars, return the string
    corresponding to the one-hot encoded tensor. tensor is expected to be of
    shape (seq_len, num_chars)

    Args
    ----------
    `s`: str
        Text to to be embedded
    `idx_to_char`: dict
        Mapping from chars to indices.
    `check_normalization`: bool, default = False
        Verify the idx_to_char mapping has the proper form.

    Returns
    ----------
    `text`: str
        Decoded string.
    """
    if check_normalization:
        blank_check = idx_to_char.get(BLANK_IDX, None)
        assert (
            blank_check == " "
        ), f"{BLANK_IDX} should map to ' ', not {blank_check}, in idx_to_char."
    text = "".join(idx_to_char.get(c.argmax().item(), "<UNK>") for c in tensor)
    return text
