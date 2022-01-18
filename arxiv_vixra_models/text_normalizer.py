from unicodedata import category

from unidecode import unidecode

# Filter ascii chars by type.
ascii_all_lower = set(chr(i).lower() for i in range(128))
ascii_control = set(ch for ch in ascii_all_lower if category(ch)[0] == "C")
ascii_all_lower_alpha_num = {ch for ch in ascii_all_lower if ch.isalnum()}
ascii_punctuation = ascii_all_lower - ascii_control - ascii_all_lower_alpha_num


def ascii_lower_char_normalizer(ch: str) -> str:
    """Normalize lower case ASCII characters.

    Description
    ----------
    Places spaces around non-alpha-numeric characters, returns lower-case
    alpha-numeric ASCII characters as they are, and maps all other characters
    to blank space.

    Parameters
    ----------
    ch : str
        Lower-case character to-be-normalized.

    Returns
    ----------
    Normalized character.
    """
    if ch in ascii_all_lower_alpha_num:
        return ch
    if ch in ascii_punctuation:
        return f" {ch} "
    # All control characters mapped to blank space.
    return " "


def text_normalizer(s: str) -> str:
    """Brutal normalization of text to ASCII.

    Description
    ----------
    Strips, lowers, and normalizes text brutally using the unidecode package.
    All remaining text is in a subset of ASCII.

    Parameters
    ----------
    s : str
        Text to-be-normalized.

    Returns
    ----------
    Normalized text.
    """
    s = s.strip()
    s = unidecode(s).lower()
    # Important to strip after ascii_lower_char_normalizer and avoid trailing
    # whitespace.
    s = "".join(ascii_lower_char_normalizer(ch) for ch in s).strip()
    # Split and join to remove multiple white spaces which occur between
    # consecutive non-alpha-numerical chars which are also not in ascii_control.
    s = " ".join(s.split())
    return s
