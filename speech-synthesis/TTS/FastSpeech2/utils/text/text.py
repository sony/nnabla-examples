import re
import unicodedata
from string import punctuation

from g2p_en import G2p

from .wdict import ___SYMBOL_TO_ID___, __ABBREVIATIONS__


def cleaner(text):
    r"""Clean text.

    Args:
        text (str): Input text.

    Returns:
        str: Text after being cleaned.
    """

    # remove accents
    text = ''.join(ch for ch in unicodedata.normalize('NFD', text)
                   if unicodedata.category(ch) != 'Mn')
    # to lowercase
    text = text.lower()

    # expand abbreviation
    for abbreviation, replacement in __ABBREVIATIONS__:
        regex = re.compile('\\b%s\\.' % abbreviation)
        text = re.sub(regex, replacement, text)

    # collapse white space
    text = re.sub(re.compile(r'\s+'), ' ', text)

    # remove trailing characters
    text = text.strip().rstrip(punctuation)

    return text


def text_to_phonemes(text):
    r"""Clean text and convert it to phonemes.

    Args:
        text (str): Input text.
    Returns:
        List[str]: List of phonemes.
    """
    g2p = G2p()  # phonemes converter
    phones = list()
    text = cleaner(text)  # cleaning text
    words = re.split(r"([,;.\-\?\!\s+])", text)

    for w in words:
        phones += list(filter(lambda p: p != " ", g2p(w)))

    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    phones = phones[1:-1].split()

    return phones


def phonemes_to_ids(phonemes):
    r"""Converting a list of phonemes to a list of IDs corresponding to the
    symbols in the text.

    Args:
        phonemes (List[str]): A list of phonemes.

    Returns:
        List[int]: List of IDs.
    """
    return [___SYMBOL_TO_ID___[p] for p in phonemes if p in ___SYMBOL_TO_ID___]
