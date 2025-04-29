import json
import logging
from pathlib import Path
from typing import Dict, Set

logger = logging.getLogger(__name__)

# default values if not specified in language config
DEFAULT_MIN_WORDS_PER_LINE = 3
DEFAULT_EOL_PUNCTUATION = {".", "?", "!", '"'}
DEFAULT_ISO639_1 = "en"
DEFAULT_ALPHANUM_REGEX = r"\p{L}|\p{N}"

# languages that don't use spaces to delimit words
SPACELESS_LANGUAGES = {"zho"}


def get_language_config(language: str) -> Dict:
    """
    Loads language-specific configuration from a JSON file, including:
        - min_words_per_line: Minimum number of words per line to consider valid
        - eol_punctuation: Set of punctuation characters considered valid at end of sentence
        - iso639_1: FastText-compatible 2-letter code
        - alphanum_regex: Unicode-aware regex pattern for checking presence of alphanumeric characters

    Falls back to English or hardcoded defaults if language is missing.
    """
    config_file = Path(__file__).parent / "../data/language_config.json"

    try:
        with config_file.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        logger.exception("Failed to load language config file.")
        config_data = {}

    lang_config = config_data.get(language) or config_data.get("en") or {}

    return {
        "min_words_per_line": lang_config.get("min_words_per_line", DEFAULT_MIN_WORDS_PER_LINE),
        "eol_punctuation": set(lang_config.get("eol_punctuation", DEFAULT_EOL_PUNCTUATION)),
        "iso639_1": lang_config.get("iso639_1", DEFAULT_ISO639_1),
        "alphanum_regex": lang_config.get("alphanum_regex", DEFAULT_ALPHANUM_REGEX),
    }


def get_supported_languages() -> Set[str]:
    """
    Returns the set of supported languages, based on the keys in the language config file.
    """
    config_file = Path(__file__).parent / "../data/language_config.json"

    try:
        with config_file.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        logger.exception("Failed to load language config file.")
        config_data = {}

    supported = set(config_data.keys())
    supported.add("agnostic")
    return supported


def get_spaceless_languages() -> Set[str]:
    """
    Returns the set of languages that do not use spaces to separate words (e.g. Chinese).
    """
    return SPACELESS_LANGUAGES