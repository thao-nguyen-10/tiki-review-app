import re, os, unicodedata, random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from underthesea import word_tokenize

def load_vietnamese_stopwords(filepath):
    """
    Load Vietnamese stopwords from a text file (one word per line).
    Returns a set of stopwords for fast lookup.
    """
    stopwords = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:  # skip empty lines
                stopwords.add(word)
    return stopwords

stopwords_path = "src/vietnamese-stopwords.txt" 
VN_STOPWORDS = load_vietnamese_stopwords(stopwords_path)

_EMOJI_PATTERN = re.compile(
    "["                     # general emojis & pictographs ranges
    "\U0001F600-\U0001F64F" # emoticons
    "\U0001F300-\U0001F5FF" # symbols & pictographs
    "\U0001F680-\U0001F6FF" # transport & map symbols
    "\U0001F1E0-\U0001F1FF" # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

_SPECIAL_CHARS_PATTERN = re.compile(r"[^0-9a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                                    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
                                    r"ùúụủũưừứựửữỳýỵỷỹđ\s]", flags=re.IGNORECASE)

MULTISPACE = re.compile(r"\s+")

def strip_emoji(text: str) -> str:
    return _EMOJI_PATTERN.sub(" ", text)

def strip_special(text: str) -> str:
    text = _SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text

def normalize_unicode(text: str) -> str:
    # NFC tends to be friendlier for Vietnamese diacritics
    return unicodedata.normalize("NFC", text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in VN_STOPWORDS and len(t) > 0]

def clean_text(text: str) -> str:
    text = str(text)
    text = normalize_unicode(text.lower())
    text = strip_emoji(text)
    text = strip_special(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)