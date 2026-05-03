import re

import contractions

# Attempt to import NLTK components, but provide safe fallbacks if resources
# are unavailable (common on hosted platforms without preinstalled NLTK data).
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    _nltk_available = True
except Exception:
    _nltk_available = False

# Prepare lemmatizer and stop words with fallbacks
if _nltk_available:
    try:
        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words('english'))
    except Exception:
        _lemmatizer = None
        _stop_words = None
else:
    _lemmatizer = None
    _stop_words = None

# Minimal fallback stop-word list (used if NLTK stopwords not available)
_FALLBACK_STOPWORDS = {
    'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'that', 'this', 'for', 'on', 'with', 'as', 'are',
    'was', 'were', 'be', 'by', 'or', 'from', 'at', 'your', 'you', 'i', 'we', 'they', 'he', 'she', 'them'
}


def _tokenize(text: str):
    if _nltk_available:
        try:
            return word_tokenize(text)
        except LookupError:
            pass
    # Basic regex tokenizer fallback
    return re.findall(r"\b\w+\b", text)


def _lemmatize(word: str):
    if _lemmatizer is not None:
        try:
            return _lemmatizer.lemmatize(word)
        except Exception:
            return word
    return word


def _get_stopwords():
    if _stop_words is not None:
        try:
            return _stop_words
        except Exception:
            return _FALLBACK_STOPWORDS
    return _FALLBACK_STOPWORDS


def clean_text(text: str) -> str:
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    tokens = _tokenize(text)
    stop_words = _get_stopwords()
    tokens = [_lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)
