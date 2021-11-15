from . import *

# Filter ascii chars by type
ascii_all_lower = set(chr(i).lower() for i in range(128))
ascii_control = set(ch for ch in ascii_all_lower if unicodedata.category(ch)[0]=='C')
ascii_all_lower_alpha_num = {ch for ch in ascii_all_lower if ch.isalnum()}
ascii_punctuation = ascii_all_lower - ascii_control - ascii_all_lower_alpha_num

def ascii_lower_char_normalizer(ch):
    """Normalizing lower case ascii characters.  Throw away all control
    characters and places spaces around other non-alpha-numeric characters.
    """
    if ch in ascii_all_lower_alpha_num:
        return ch
    # Spaces around punctutation chars.
    if ch in ascii_punctuation:
        return f' {ch} '
    # All control characters mapped to blank space.
    return ' '

def text_normalizer(s):
    """Brutal normalization of text to ASCII."""
    s = s.strip()
    s = unidecode(s).lower()
    # Important to strip after ascii_lower_char_normalizer and avoid trailing
    # whitespace.
    s = ''.join(ascii_lower_char_normalizer(ch) for ch in s).strip()
    # Split and join to remove multiple white spaces which occur between
    # consecutive non-alpha-numerical chars which are also not in ascii_control.
    s = ' '.join(s.split())
    return s